/*****************************************************************
 This file is part of monocular camera calibration library.
 Copyright(C) 2018-2020, Zhilong (Dgelom) Su, all rights reserved.
******************************************************************/
#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\calib3d.hpp>
#include <Matrice\core\matrix.h>
#include <Matrice\core\solver.h>
#include <Matrice\arch\ixpacket.h>
#include <Matrice\algs\geometry\transform.h>
#include <Matrice\algs\geometry\normalization.h>
#include "../../include/calibration.hpp"
#include "../cost/cost_functions.hpp"

namespace dgelom {
namespace detail {
#define _Myt _Calibrator<_Ty, _Pt>

template<typename _Ty, pattern_type _Pt>
_Calibrator<_Ty, _Pt>::_Calibrator(const pattern_t& _Pattern)
	:m_pattern(_Pattern) {
	_Get_planar_points();
}

template<typename _Ty, pattern_type _Pt>
_Calibrator<_Ty, _Pt>::_Calibrator(const image_info_t& _Fnames, const pattern_t& _Pattern)
	: m_fnames(_Fnames), m_pattern(_Pattern) {
	_Get_planar_points();
	_Get_image_points();
}

template<typename _Ty, pattern_type _Pt>
typename _Myt::plane_array& _Calibrator<_Ty, _Pt>::run(bool require_optim) {
	_Analysis();
	if (require_optim) {
		_Optimize();
	}
	_Eval_true_to_scale();
	return m_params;
}

template<typename _Ty, pattern_type _Pt>
const typename _Myt::matrix_t& _Calibrator<_Ty, _Pt>::get_poses() const noexcept
{
	return (m_poses);
}

template<typename _Ty, pattern_type _Pt>
const typename _Myt::ptarray_t&_Calibrator<_Ty,_Pt>::planar_points()const noexcept
{
	return (m_points);
}

template<typename _Ty, pattern_type _Pt>
typename _Myt::ptarray_t& _Calibrator<_Ty, _Pt>::planar_points() noexcept
{
	return m_points;
}

template<typename _Ty, pattern_type _Pt>
const typename _Myt::ptarray_t& _Calibrator<_Ty, _Pt>::image_points() const noexcept
{
	return m_imgpts;
}

template<typename _Ty, pattern_type _Pt>
typename _Myt::ptarray_t& _Calibrator<_Ty, _Pt>::image_points() noexcept
{
	return m_imgpts;
}

template<typename _Ty, pattern_type _Pt>
void _Calibrator<_Ty, _Pt>::_Get_image_points() {
	using namespace cv;
#ifdef MATRICE_DEBUG
	std::cout << " >> Camera calibrator detects corners: \n";
#endif
	const auto _Size = m_pattern.size<Size>();
	const auto _Nimgs = m_fnames.count();
	const int  _N = m_pattern.count();

	m_effindices.clear();
	std::vector<std::vector<Point2f>> _Impts(_Nimgs);

	for (index_t i = 0; i < _Nimgs; ++i) {
#ifdef MATRICE_DEBUG
	std::cout << "	Processing image [" << i + 1 << "/" << _Nimgs << "]: ";
#endif
		auto _Src = cv::imread(m_fnames(i), 0);
		m_iw = _Src.cols, m_ih = _Src.rows;

		bool _Found = false;
		std::vector<Point2f> _Points;
		if constexpr (m_pattern.type == pattern_type::squared)
			_Found = findChessboardCorners(_Src, _Size, _Points,
				CALIB_CB_ADAPTIVE_THRESH +
				CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		if constexpr (m_pattern.type == pattern_type::circular)
			_Found = findCirclesGrid(_Src, _Size, _Points,
				CALIB_CB_SYMMETRIC_GRID);
		if (_Found) {
			if constexpr (m_pattern.type == pattern_type::squared)
				cornerSubPix(_Src, _Points, Size(11, 11), Size(-1, -1),
					TermCriteria(criteria_t::COUNT + criteria_t::EPS, 30, 0.1));

			_Impts[i] = _Points;

			m_effindices.push_back(static_cast<size_t>(i));

#ifdef MATRICE_DEBUG
		std::cout << m_pattern.cols() << " x " << m_pattern.rows() << " = " << _Points.size() << " corners\n";
#endif
		}
	}

	m_imgpts.create(m_effindices.size() << 1, _N);
	for (auto i = 0; i < m_effindices.size(); ++i) {
		const auto _Points = _Impts[m_effindices[i]];
		auto _Px = m_imgpts[i << 1], _Py = m_imgpts[i << 1 | 1];
		size_t _Pos = 0;
		for_each(_Points, [&](const auto& _Point) {
			_Px[_Pos] = _Point.x; _Py[_Pos++] = _Point.y;
			});
	}

	m_normal = { 1. / m_iw, 0., -0.5, 0., 1. / m_ih, -0.5, 0., 0., -1. };
}

template<typename _Ty, pattern_type _Pt>
void _Calibrator<_Ty, _Pt>::_Get_planar_points() {
	const auto[_Cols, _Rows] = m_pattern.size<std::pair<size_t, size_t>>();
	const auto _Pitch = m_pattern.pitch();
	m_points.create(2, m_pattern.count());
	auto _Rowx = m_points.ptr(0), _Rowy = m_points.ptr(1);
	for (size_t r = 0; r < _Rows; ++r) {
		value_t _Cooy = r * _Pitch;
		for (size_t c = 0; c < _Cols; ++c) {
			value_t _Coox = c * _Pitch;
			auto _Idx = r * _Cols + c;
			_Rowx[_Idx] = _Coox, _Rowy[_Idx] = _Cooy;
		}
	}
}

template<typename _Ty, pattern_type _Pt>
void _Calibrator<_Ty, _Pt>::_Eval_true_to_scale() {
	const auto _Rows = m_pattern.rows(), _Cols = m_pattern.cols();
	const auto _S = 1.0 / sq(m_pattern.pitch());
	
	Vec3_<value_t> X{ 0, 0, 0 };
	auto R = matrix3x3::zeros();
	for (auto i = 0; i < m_poses.rows(); ++i) {
		const auto rt = m_poses[i];
		Vec3_<value_t> r{ rt[0], rt[1], rt[2] };
		rodrigues(r, R);
		auto u = R.mul(X) + decltype(r){rt[3], rt[4], rt[5]};
	}
}

template<typename _Ty, pattern_type _Pt>
typename _Myt::ptarray_t _Calibrator<_Ty, _Pt>::_Normalize(size_t i)
{
	using packed_t = simd::template Packet_<ptarray_t::value_t, 4>;

	ptarray_t _Normalized(2, m_imgpts.cols());

	packed_t _Unit(0.5), _Normx(m_normal(0)), _Normy(m_normal(4));
	auto _Pos_x = m_imgpts[i << 1], _Pos_y = m_imgpts[i << 1 | 1];

	for (int i = 0; i < m_imgpts.cols(); i += packed_t::size) {
		(packed_t(_Pos_x + i)*_Normx - _Unit).unpack(_Normalized[0] + i);
		(packed_t(_Pos_y + i)*_Normy - _Unit).unpack(_Normalized[1] + i);
	}

	auto ii = m_imgpts.cols() - m_imgpts.cols() / packed_t::size;
	(packed_t(_Pos_x + ii)*_Normx - _Unit).unpack(_Normalized[0] + ii);
	(packed_t(_Pos_y + ii)*_Normy - _Unit).unpack(_Normalized[1] + ii);

	return move(_Normalized);
}

template<typename _Ty, pattern_type _Pt>
typename _Myt::matrix3x3 _Calibrator<_Ty, _Pt>::_Find_homography(size_t i)
{
	auto _Points = this->_Normalize(i);

	size_t _Npts = _Points.cols();
	auto _Img_x = _Points[0], _Img_y = _Points[1];
	auto _Pla_x = m_points[0], _Pla_y = m_points[1];

	matrix_t _L(_Npts << 1, 9);
	for (size_t i = 0; i < _Npts; ++i) {
		auto x = _Img_x[i], y = _Img_y[i];
		auto X = _Pla_x[i], Y = _Pla_y[i];

		auto c1 = i << 1;
		_L[c1][0] = X, _L[c1][1] = Y, _L[c1][2] = 1.;
		_L[c1][3] = 0., _L[c1][4] = 0., _L[c1][5] = 0.;
		_L[c1][6] = -x * X, _L[c1][7] = -x * Y, _L[c1][8] = -x;

		auto c2 = i << 1 | 1;
		_L[c2][0] = 0., _L[c2][1] = 0., _L[c2][2] = 0.;
		_L[c2][3] = X, _L[c2][4] = Y, _L[c2][5] = 1.;
		_L[c2][6] = -y * X, _L[c2][7] = -y * Y, _L[c2][8] = -y;
	}

	using Op_t = detail::LinearOp::Svd<decltype(_L)>;
	typename detail::Solver::Linear_<Op_t> solver(_L);
	auto _Ret = solver.solve();
	auto _Nor = _Ret.normalize(_Ret(8));

	return matrix3x3({
		_Nor(0), _Nor(1), _Nor(2),
		_Nor(3), _Nor(4), _Nor(5),
		_Nor(6), _Nor(7), value_t(1) });
}

template<typename _Ty, pattern_type _Pt>
typename _Myt::plane_array& _Calibrator<_Ty, _Pt>::_Analysis()
{
	enum { _Elems = Npose, };
	using matrix6x1 = Matrix_<value_t, _Elems, compile_time_size<>::val_1>;
	using packed_t = simd::Packet_<value_t, 4>;

	m_poses.create(m_effindices.size(), _Elems);
	std::vector<matrix3x3> _Homo_buf(m_poses.rows());
	matrix_t _V(m_poses.rows() << 1, _Elems);

	for (size_t i = 0; i < m_poses.rows(); ++i) {
		matrix3x3 _H = _Find_homography(i);

		value_t h11 = _H[0][0], h12 = _H[1][0], h13 = _H[2][0];
		value_t h21 = _H[0][1], h22 = _H[1][1], h23 = _H[2][1];
		value_t h31 = _H[0][2], h32 = _H[1][2], h33 = _H[2][2];

		_Homo_buf[i] = _H;
		matrix6x1 _Temp1{ h11 * h21, h11 * h22 + h12 * h21, h12 * h22,
			h13 * h21 + h11 * h23, h13 * h22 + h12 * h23, h13 * h23 };
		auto _Norm_temp1 = _Temp1.normalize(_Temp1.norm_2());

		matrix6x1 _Temp2{ h11 * h11 - h21 * h21, (h11 * h12 - h21 * h22) * 2,
			h12 * h12 - h22 * h22, (h13 * h11 - h21 * h23) * 2,
			(h13 * h12 - h23 * h22) * 2, h13 * h13 - h23 * h23 };
		auto _Norm_temp2 = _Temp2.normalize(_Temp2.norm_2());

		auto _Begin1 = _V[i << 1], _Begin2 = _V[i << 1 | 1];
		for (size_t j = 0; j < _Elems; ++j) {
			_Begin1[j] = _Norm_temp1(j);
			_Begin2[j] = _Norm_temp2(j);
		}
	}

	using Op_t = detail::LinearOp::Svd<decltype(_V)>;
	typename detail::Solver::Linear_<Op_t> solver(_V);
	auto _Ret = solver.solve();

	const value_t B11 = _Ret(0), B12 = _Ret(1), B13 = _Ret(3);
	const value_t B22 = _Ret(2), B23 = _Ret(4), B33 = _Ret(5);
	const value_t _Val1 = B12 * B13 - B11 * B23;
	const value_t _Val2 = B11 * B22 - B12 * B12;
#ifdef MATRICE_DEBUG
	DGELOM_CHECK(abs(_Val2) > matrix_t::eps && abs(B11) > matrix_t::eps,
		"Unstable numeric computation.");
#endif

	const value_t v0 = _Val1 / _Val2;
	const value_t lambda = B33 - (B13 * B13 + v0 * _Val1) / B11;
	const value_t alpha = sqrt(lambda / B11);
	const value_t beta = sqrt(lambda * B11 / _Val2);
	const value_t gamma = -B12 * beta / B11;
	const value_t u0 = -(B12 * v0 + B13) / B11;

	matrix3x3 _Normal_inv{ 1 / m_normal(0), 0., -m_normal(2) / m_normal(0),
		0., 1 / m_normal(4), -m_normal(5) / m_normal(4), 0.,0.,1. };
	auto _Ex = _Normal_inv.mul(matrix3x3{ alpha, gamma, u0, 0., beta, v0, 0., 0., 1. });
	const auto fx = m_params(0) = _Ex(0), fy = m_params(1) = _Ex(4);
	const auto cx = m_params(2) = _Ex(2), cy = m_params(3) = _Ex(5);
	const auto fs = m_params(6) = _Ex(1)/100;
	m_params(4) = m_params(5) = 0.001;

	const auto rows = m_effindices.size()*m_pattern.count() << 1;
	Matrix_<value_t, ::dynamic, 2> A(rows);
	Matrix_<value_t, ::dynamic, 1> b(rows);

	matrix3x3 _K_inv{ 1 / fx, -fs / (fx*fy), (cy*fs / fy - cx) / fx, 0, 1 / fy, -cy / fy, 0, 0, 1 };
	for (size_t i = 0; i < m_poses.rows(); ++i) {
		const auto& _H_view = _Homo_buf[i];
		auto _Ex_r = _K_inv.mul(_Normal_inv.mul(_H_view).eval<matrix3x3>());

		Vec3_<value_t> _R1{ _Ex_r(0,0), _Ex_r(1,0), _Ex_r(2,0) };
		Vec3_<value_t> _R2{ _Ex_r(0,1), _Ex_r(1,1), _Ex_r(2,1) };
		value_t _Lambda[] = { 1 / _R1.norm_2(), 1 / _R2.norm_2(), 0. };
		_R1 = _Lambda[0] * _R1, _R2 = _Lambda[1] * _R2;

		auto _Begin = m_poses[i];

		_Lambda[2] = (_Lambda[0] + _Lambda[1]) / 2;
		(_Lambda[2] * packed_t{ _Ex_r(0,2), _Ex_r(1,2), _Ex_r(2,2) }).unpack(_Begin + 3);

		decltype(_R1) _R3 = _R1.cross(_R2);
		matrix3x3 _R{ _R1.x, _R2.x, _R3.x, _R1.y, _R2.y, _R3.y, _R1.z, _R2.z, _R3.z };
		detail::LinearOp::Svd<decltype(_R)> _Op(_R);
		decltype(_R) _R_opt = _R.mul(_Op.vt());
		rodrigues(_R_opt, _Begin);

		_Update_dist_eqs(i, A, b, _R_opt, _Begin);
	}

	auto AtAi = A.t().mul(A).eval().inv().eval();
	auto Atb = A.t().mul(b).eval();
	auto k = AtAi.mul(Atb);
	m_params(4) = k(0), m_params(5) = k(1);

	return (m_params);
}

template<typename _Ty, pattern_type _Pt>
void _Calibrator<_Ty, _Pt>::_Update_dist_eqs(size_t i, Matrix_<value_t, ::dynamic, 2>& A, Matrix_<value_t, ::dynamic, 1>& b, const matrix3x3 & R, const value_t * T)
{
	const auto pt = m_imgpts.block<::extent_x>(i << 1, 2);
	const auto fx = m_params(0), fy = m_params(1);
	const auto cx = m_params(2), cy = m_params(3);
	const auto fs = m_params(6);
	for (auto c = 0; c < m_imgpts.cols(); ++c) {
		const auto u = pt[0][c], v = pt[1][c];
		const auto x_d = (u - cx) / fx, y_d = (v - cy) / fy;

		Vec3_<value_t> X(m_points[0][c], m_points[1][c], 0);
		decltype(X) X_c = R.mul(X) + decltype(X)(T[3], T[4], T[5]);
		const auto x_i = X_c.x / X_c.z, y_i = X_c.y / X_c.z;
		const auto u_i = fx * x_i + cx, v_i = fy * y_i + cy;

		const auto j = i * m_imgpts.cols() + c;
		if constexpr (Dmodel == distortion_model::D2U) {
			const auto s = sq(x_d) + sq(y_d);
			A[j << 1][0] = (u - cx)*s, A[j << 1][1] = (u - cx)*sq(s);
			A[j << 1 | 1][0] = (v - cy)*s, A[j << 1 | 1][1] = (v - cy)*sq(s);
			b(j << 1) = u_i - u, b(j << 1 | 1) = v_i - v;
		}
		else {
			const auto s = sq(x_i) + sq(y_i);
			A[j << 1][0] = (u_i - cx)*s, A[j << 1][1] = (u_i - cx)*sq(s);
			A[j << 1 | 1][0] = (v_i - cy)*s, A[j << 1 | 1][1] = (v_i - cy)*sq(s);
			b(j << 1) = u - u_i, b(j << 1 | 1) = v - v_i;
		}
	}
}

template<typename _Ty, pattern_type _Pt>
void _Calibrator<_Ty, _Pt>::_Optimize()
{
	const auto _Nviews = this->m_effindices.size();
	const auto _Nctrls = this->m_points.cols();

	auto &_Fx = m_params(0), &_Fy = m_params(1);
	auto &_Cx = m_params(2), &_Cy = m_params(3);
	const auto _K1 = m_params(4), _K2 = m_params(5);
	const auto _Fs = m_params(6);

	Matrix<default_type> _Plane_points(_Nctrls, 3);
	transform(m_points[0], m_points[0] + _Nctrls, _Plane_points.begin(), 3);
	transform(m_points[1], m_points[1] + _Nctrls, _Plane_points.begin() + 1, 3);
	fill(_Plane_points.begin() + 2, _Plane_points.end(), 3, default_type(0));

	Matrix_<default_type, 7, 1> _Vars = 1;
	//_Vars(4) = _Vars(5) = 0.001;
	auto _Problem = std::make_shared<xop::Problem>();
	for (auto _Idx = 0; _Idx < _Nviews; ++_Idx) {
		auto _Row = m_effindices[_Idx] << 1;
		auto _Pose = m_poses.ptr(_Idx);
		for (index_t i = 0; i < _Nctrls; ++i) {
			const auto _U0 = m_imgpts[_Row][i];
			const auto _V0 = m_imgpts[_Row + 1][i];
			const Vec_<default_type, 2> _Point{ _U0, _V0 };
			auto _Cost = CostFunc::_Std_mono_error<decltype(_Point), Dmodel>
				::Create(_Fx, _Fy, _Cx, _Cy, _Point);
			_Problem->AddResidualBlock(_Cost, new xop::HuberLoss(0.05), _Vars.begin(), _Pose, _Plane_points[i]);
		}
	}

	xop::Solver::Options _Options;
	_Options.linear_solver_type = xop::DENSE_SCHUR;
	_Options.max_num_iterations = 500;
	_Options.num_threads = 8;
#ifdef MATRICE_DEBUG
	_Options.minimizer_progress_to_stdout = true;
#endif

	auto _Summary = std::make_unique<xop::Solver::Summary>();
	xop::Solve(_Options, _Problem.get(), _Summary.get());
#ifdef MATRICE_DEBUG
	std::cout << _Summary->FullReport() << std::endl;
#endif
	m_params = m_params * _Vars;

	m_error = _Reprojection_error(_Problem);
}


template class _Calibrator<double, pattern_type::squared>;
template class _Calibrator<double, pattern_type::circular>;
}
}