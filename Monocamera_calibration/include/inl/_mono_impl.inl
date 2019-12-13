/***************************************************************************
This file is part of monocular camera calibration library.
Copyright(C) 2018-2020, Zhilong (Dgelom) Su, all rights reserved.

This program is free software : you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
****************************************************************************/
#pragma once
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
#include "..\mono_calibrator.h"

namespace dgelom {
template<typename T, pattern_type _Patt, distortion_model _Order>
INLINE void mono_calibrator<T, _Patt, _Order>::_Retrieve_from_bg() {
}

template<typename T, pattern_type _Patt, distortion_model _Order>
INLINE void mono_calibrator<T, _Patt, _Order>::_Get_image_points() {
	using namespace cv;

#ifdef MATRICE_DEBUG
	std::cout << " >> Camera calibrator detects corners: \n";
#endif
		const auto _Size = m_pattern.size<Size>();
		const auto _Nimgs = m_fnames.count()/8;
		const int  _N = m_pattern.count();

		m_imgpts.create(_Nimgs << 1, _N);
		m_effindices.clear();
#pragma omp parallel 
{
#pragma omp for
		for (index_t i = 0; i < _Nimgs; ++i) {
#ifdef MATRICE_DEBUG
			std::cout << "	Processing image [" << i+1 << "/"<<_Nimgs<<"]: ";
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
				
				auto _Px = m_imgpts[i << 1], _Py = m_imgpts[i << 1|1];
				size_t _Pos = 0;
				for_each(_Points, [&](const auto& _Point) {
					_Px[_Pos] = _Point.x; _Py[_Pos++] = _Point.y;
				});
				m_effindices.push_back(static_cast<size_t>(i));
#ifdef MATRICE_DEBUG
				std::cout << m_pattern.cols() << " x " << m_pattern.rows() << " = " << _Points.size() << " corners\n";
#endif
			}
		}
		m_normal = { 2. / m_iw, 0., -1., 0., 2. / m_ih, -1, 0., 0., -1. };
}
}

template<typename T, pattern_type _Patt, distortion_model _Order>
INLINE void mono_calibrator<T, _Patt, _Order>::_Get_planar_points() {
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

template<typename T, pattern_type _Patt, distortion_model _Order>
INLINE typename mono_calibrator<T, _Patt, _Order>::ptarray_t
mono_calibrator<T, _Patt, _Order>::_Normalize(size_t i) {
	using packed_t = simd::template Packet_<ptarray_t::value_t, 4>;

	ptarray_t _Normalized(2, m_imgpts.cols());

	packed_t _Unit(1.), _Normx(m_normal(0)), _Normy(m_normal(4));
	auto _Pos_x = m_imgpts[i<<1], _Pos_y = m_imgpts[i<<1|1];

	for (int i = 0; i < m_imgpts.cols(); i += packed_t::size) {
		(packed_t(_Pos_x + i)*_Normx - _Unit).unpack(_Normalized[0] + i);
		(packed_t(_Pos_y + i)*_Normy - _Unit).unpack(_Normalized[1] + i);
	}

	auto ii = m_imgpts.cols() - m_imgpts.cols() / packed_t::size;
	(packed_t(_Pos_x + ii)*_Normx - _Unit).unpack(_Normalized[0] + ii);
	(packed_t(_Pos_y + ii)*_Normy - _Unit).unpack(_Normalized[1] + ii);

	return (_Normalized);
}

template<typename T, pattern_type _Patt, distortion_model _Order>
INLINE void mono_calibrator<T, _Patt, _Order>::_Get_true_to_scale() {
	/*_My_future_scale = std::async(std::launch::async, [&] {
		auto _Rows = m_pattern.rows(), _Cols = m_pattern.cols();
		auto _S = 1.0/std::pow(m_pattern.pitch(), 2);
		std::vector<matrix_t::value_t> _Scale;

		for (const auto& _Points : m_ipoints) {
			matrix_t _Dx(_Points.cols(), 1);
			for (size_t r = 0; r < _Rows; ++r) {
				const auto _Begin = _Points[0] + r * _Cols;
				const auto _End = _Begin + _Cols;
				std::adjacent_difference(_Begin, _End, _Dx.begin() + r * _Cols);
			}
			matrix_t _Dy(_Points.cols(), 1);
			for (size_t r = 0; r < _Rows; ++r) {
				const auto _Begin = _Points[1] + r * _Cols;
				const auto _End = _Begin + _Cols;
				std::adjacent_difference(_Begin, _End, _Dy.begin()+ r * _Cols);
			}

			matrix_t _D = _S*(_Dx*_Dx + _Dy*_Dy);
			_Scale.push_back(reduce(_D.begin() + 1, _D.end(), _Cols));
		}
		m_scale = matrix_t(_Scale.size(), 1, _Scale.data()).sum()/_Scale.size()/(_Cols-1)/_Rows;
	});*/
}

template<typename T, pattern_type _Patt, distortion_model _Order>
INLINE typename mono_calibrator<T, _Patt, _Order>::matrix3x3 
mono_calibrator<T, _Patt, _Order>::_Find_homography(size_t i) {
	decltype(auto) _Points = this->_Normalize(i);

	size_t _Npts = _Points.cols();
	auto _Img_x = _Points[0],  _Img_y = _Points[1];
	auto _Pla_x = m_points[0], _Pla_y = m_points[1];

	matrix_t _L(_Npts << 1, 9);
	for (size_t i = 0; i < _Npts; ++i) {
		auto x = _Img_x[i], y = _Img_y[i];
		auto X = _Pla_x[i], Y = _Pla_y[i];

		auto c1 = i << 1;
		_L[c1][0] = X,      _L[c1][1] = Y,      _L[c1][2] = 1.;
		_L[c1][3] = 0.,     _L[c1][4] = 0.,     _L[c1][5] = 0.;
		_L[c1][6] = -x * X, _L[c1][7] = -x * Y, _L[c1][8] = -x;

		auto c2 = i << 1|1;
		_L[c2][0] = 0.,     _L[c2][1] = 0.,     _L[c2][2] = 0.;
		_L[c2][3] = X,      _L[c2][4] = Y,      _L[c2][5] = 1.;
		_L[c2][6] = -y * X, _L[c2][7] = -y * Y, _L[c2][8] = -y;
	}
	
	using Op_t = detail::LinearOp::Svd<decltype(_L)>;
	typename types::Solver::Linear_<Op_t> solver(_L);
	auto _Ret = solver.solve();
	auto _Nor = _Ret.normalize(_Ret(8));

	return matrix3x3({ 
		_Nor(0), _Nor(1), _Nor(2), 
		_Nor(3), _Nor(4), _Nor(5), 
		_Nor(6), _Nor(7), value_t(1) });
}

template<typename T, pattern_type _Patt, distortion_model _Order>
INLINE typename mono_calibrator<T, _Patt, _Order>::plane_array& 
mono_calibrator<T, _Patt, _Order>::_Analysis() {
	enum { _Elems = Npose, };
	using matrix6x1 = Matrix_<value_t, _Elems, compile_time_size<>::val_1>;
	using packed_t = simd::Packet_<value_t, 4>;
	
	_Get_true_to_scale();
	m_poses.create(m_effindices.size(), _Elems);
	std::vector<matrix3x3> _Homo_buf(m_poses.rows());
	matrix_t _V(m_poses.rows() << 1, _Elems);
	_Retrieve_from_bg();

	for (size_t i = 0; i < m_poses.rows(); ++i) {
		matrix3x3 _H = _Find_homography(m_effindices[i]);

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
	typename types::Solver::Linear_<Op_t> solver(_V);
	auto _Ret = solver.solve();
	
	value_t B11 = _Ret(0), B12 = _Ret(1), B13 = _Ret(3);
	value_t B22 = _Ret(2), B23 = _Ret(4), B33 = _Ret(5);
	value_t _Val1 = B12 * B13 - B11 * B23;
	value_t _Val2 = B11 * B22 - B12 * B12;
#ifdef MATRICE_DEBUG
	DGELOM_CHECK(abs(_Val2) > matrix_t::eps && abs(B11) > matrix_t::eps,
		"Unstable numeric computation.");
#endif

	value_t v0 = _Val1 / _Val2;                                  //cy
	value_t lambda = B33 - (B13 * B13 + v0 * _Val1) / B11;
	value_t alpha = sqrt(lambda / B11);                          //fx
	value_t beta = sqrt(lambda * B11 / _Val2);                   //fy
	value_t gamma = -B12 * beta / B11;                           //fs
	value_t u0 = -(B12 * v0 + B13) / B11;                        //cx

	matrix3x3 _Normal_inv{1/m_normal(0), 0., -m_normal(2)/ m_normal(0),
		0., 1 / m_normal(4), -m_normal(5) / m_normal(4), 0.,0.,1.};
	auto _Ex = _Normal_inv.mul(matrix3x3{alpha, gamma, u0, 0., beta, v0, 0., 0., 1.});
	auto fx = m_params(0) = _Ex(0), fy = m_params(1) = _Ex(4);
	auto cx = m_params(2) = _Ex(2), cy = m_params(3) = _Ex(5);
	auto fs = m_params(6) = _Ex(1);
	m_params(4) = m_params(5) = 0.1;

	const auto rows = m_effindices.size()*m_pattern.count() << 1;
	Matrix_<value_t, ::dynamic, 2> A(rows);
	Matrix_<value_t, ::dynamic, 1> b(rows);

	matrix3x3 _K_inv{1/fx, -fs/(fx*fy), (cy*fs/fy - cx)/fx, 0, 1/fy, -cy/fy, 0, 0, 1};
	for (size_t i = 0; i < m_poses.rows(); ++i) {
		const auto& _H_view = _Homo_buf[i];
		auto _Ex_r = _K_inv.mul(_Normal_inv.mul(_H_view).eval<matrix3x3>());

		Vec3_<value_t> _R1{ _Ex_r(0,0), _Ex_r(1,0), _Ex_r(2,0) };
		Vec3_<value_t> _R2{ _Ex_r(0,1), _Ex_r(1,1), _Ex_r(2,1) };
		value_t _Lambda[] = { 1 / _R1.norm_2(), 1 / _R2.norm_2(), 0.};
		_R1 = _Lambda[0] * _R1, _R2 = _Lambda[1] * _R2;
		
		auto _Begin = m_poses[i];

		_Lambda[2] = (_Lambda[0] + _Lambda[1]) / 2;
		(_Lambda[2]*packed_t{_Ex_r(0,2), _Ex_r(1,2), _Ex_r(2,2)}).unpack(_Begin + 3);

		decltype(_R1) _R3 = _R1.cross(_R2);
		matrix3x3 _R{ _R1.x, _R2.x, _R3.x, _R1.y, _R2.y, _R3.y, _R1.z, _R2.z, _R3.z };
		detail::LinearOp::Svd<decltype(_R)> _Op(_R);
		decltype(_R) _R_opt = _R.mul(_Op.vt());

		_Update_dist_eqs(i, A, b, _R_opt, _Begin);

		rodrigues(_R_opt, _Begin);
	}

	auto B = A.t().mul(A).eval().inv();
	const auto k = B.mul(A.t().mul(b).eval());
	m_params(4) = k(0), m_params(5) = k(1);

	return (m_params);
}

template<typename T, pattern_type _Patt, distortion_model _Model>
INLINE void mono_calibrator<T, _Patt, _Model>::_Update_dist_eqs(size_t i, Matrix_<value_t, ::dynamic, 2>& A, Matrix_<value_t, ::dynamic, 1>& b, const matrix3x3& R, const value_t* T) {
	const auto pt = m_imgpts.block<::extent_x>(i << 1, 2);
	const auto fx = m_params(0), fy = m_params(1);
	const auto cx = m_params(2), cy = m_params(3);
	for (auto c = 0; c < m_imgpts.cols(); ++c) {
		// observed image point
		const auto u = pt[0][c], v = pt[1][c];
		// distorted normalized image point 
		const auto x_d = u / fx - cx, y_d = v / fy - cy;
		// object point
		Vec3_<value_t> X(m_points[0][c], m_points[1][c], 0);
		// ideal normalized image point
		decltype(X) x = R.mul(X) + decltype(X)(T[3], T[4], T[5]);

		const auto j = i * m_imgpts.cols() + c;
		if constexpr (Dmodel == distortion_model::D2U) {
			const auto s = sqr(x_d) + sqr(y_d);
			A[j << 1][0] = x_d * s, A[j << 1][1] = x_d * sqr(s);
			A[j << 1|1][0] = y_d * s, A[j << 1|1][1] = y_d * sqr(s);
			b(j << 1) = x_d - x.x/x.z, b(j << 1 | 1) = y_d - x.y/x.z;
		}
		else {

		}
	}
}

template<typename T, pattern_type _Patt, distortion_model _Order>
INLINE typename mono_calibrator<T, _Patt, _Order>::plane_array& 
mono_calibrator<T, _Patt, _Order>::run() {
	return _Analysis();
}
}