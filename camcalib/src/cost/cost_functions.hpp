#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "../calibration/_interp.hpp"

namespace dgelom {
namespace xop = ceres;
template<typename T>
void _Rotate(const T* ang, const T x[3], T y[3]) {
	using value_t = T;
	const auto _Zero = value_t(0);
	const auto _Unit = value_t(1);

	value_t theta = xop::sqrt(xop::DotProduct(ang, ang));

	value_t c = xop::cos(theta), s = xop::sin(theta);
	value_t c1 = _Unit - c;
	value_t itheta = _Unit / theta;

	value_t r[] = { ang[0] * itheta, ang[1] * itheta, ang[2] * itheta };
	value_t R[9]{c*_Unit, _Zero, _Zero, _Zero, c*_Unit, _Zero, _Zero, _Zero, c*_Unit};

	R[0] = R[0] + c1*r[0]*r[0], R[1] = R[1] + c1*r[0]*r[1], R[2] = R[2] + c1*r[0]*r[2];
	R[3] = R[3] + c1*r[0]*r[1], R[4] = R[4] + c1*r[1]*r[1], R[5] = R[5] + c1*r[1]*r[2];
	R[6] = R[6] + c1*r[0]*r[2], R[7] = R[7] + c1*r[1]*r[2], R[8] = R[8] + c1*r[2]*r[2];

	R[1] = R[1] - s * r[2], R[2] = R[2] + s * r[1];
	R[3] = R[3] + s * r[2], R[5] = R[5] - s * r[0];
	R[6] = R[6] - s * r[1], R[7] = R[7] + s * r[0];

	y[0] = R[0] * x[0] + R[1] * x[1] + R[2] * x[2];
	y[1] = R[3] * x[0] + R[4] * x[1] + R[5] * x[2];
	y[2] = R[6] * x[0] + R[7] * x[1] + R[8] * x[2];
}

template<typename T>
void _Rigid_transform(const T* _motion_vector, const T x[3], T y[3])
{
	_Rotate(_motion_vector, x, y);

	y[0] = y[0] + _motion_vector[3];
	y[1] = y[1] + _motion_vector[4];
	y[2] = y[2] + _motion_vector[5];
}

template<typename _Prob_pointer> 
double _Reprojection_error(const _Prob_pointer _Problem)
{
	auto _Sq = [](auto _Val) {return _Val * _Val; };

	xop::Problem::EvaluateOptions _Option;
	_Problem->GetResidualBlocks(&_Option.residual_blocks);

	double _Total_cost = 0.0;
	std::vector<double> _Residuals;
	_Problem->Evaluate(_Option, &_Total_cost, &_Residuals, nullptr, nullptr);
	
	double _Error = 0;
	const auto _Size = _Option.residual_blocks.size();
	for (auto i = 0; i < _Size; ++i)
		_Error += std::sqrt(_Sq(_Residuals[i<<1])+_Sq(_Residuals[i<<1|1]));
	
	return _Error / _Size;
}

struct CostFunc {
	using value_t = default_type;
	using _InIt = value_t * ;

	template<typename _Point_type, distortion_model _Dmodel> 
	struct _Std_mono_error
	{
		using point_t = _Point_type;

		inline _Std_mono_error(value_t _Fx, value_t _Fy, value_t _Cx, value_t _Cy, const point_t& _Ipt)
			:m_fx(_Fx), m_fy(_Fy), m_cx(_Cx), m_cy(_Cy), m_pt(_Ipt) {}
		inline _Std_mono_error(value_t _Fx, value_t _Fy, value_t _Cx, value_t _Cy, const point_t& _Ipt, const point_t& _Ppt)
			: m_fx(_Fx), m_fy(_Fy), m_cx(_Cx), m_cy(_Cy), m_pt(_Ipt), m_X(_Ppt) {}

		template<typename T> __forceinline
		bool operator() (const T* const _Pars, const T* const _Pose, const T* const _X, T* _Residual) const
		{
			using jet_t = T;
#define  aligned_jet_t alignas(32) jet_t
			const auto _Dot_one = jet_t(0.1);

			jet_t _Fx = m_fx * _Pars[0], _Fy = m_fy * _Pars[1], _Fs = _Pars[6];
			jet_t _Cx = m_cx * _Pars[2], _Cy = m_cy * _Pars[3];
			jet_t _K1 = _Dot_one * _Pars[4], _K2 = _Dot_one * _Pars[5];

			aligned_jet_t _Nx[3]; _Rigid_transform(_Pose, _X, _Nx);
			_Nx[0] = _Nx[0] / _Nx[2];
			_Nx[1] = _Nx[1] / _Nx[2];

			auto _U0 = jet_t(m_pt[0]), _V0 = jet_t(m_pt[1]);
			if constexpr (_Dmodel == distortion_model::D2U) {
				auto _Du = _U0 - _Cx, _Dv = _V0 - _Cy;
				auto _Dy = _Dv / _Fy;
				auto _Dx = (_Du - _Dv*_Fs / _Fy) / _Fx;

				auto _R_sq = _Dx * _Dx + _Dy * _Dy;
				auto _D = jet_t(1) + _R_sq*(_K1 + _K2*_R_sq);

				//C-1: x(fx, fy, fs, cx, cy, k1, k2; ¦Äu[optional]) - x'(r, t; X)
				auto _Ux = _D * _Dx, _Uy = _D * _Dy;
				_Residual[0] = _Ux - _Nx[0];
				_Residual[1] = _Uy - _Nx[1];

				//C-2: u_c(fx, fy, fs, cx, cy, k1, k2) - u'_c(fx, fy, fs; r, t; X)
				/*auto _Uu = _D * _Du, _Uv = _D * _Dv;
				auto _Ru = _Fx * _Nx[0] + _Fs * _Nx[1];
				auto _Rv = _Fy * _Nx[1];
				_Residual[0] = _Uu - _Ru;
				_Residual[1] = _Uv - _Rv;*/
			}
			if constexpr (_Dmodel == distortion_model::U2D) {
				//C: u - u'(fx, fy, fs, cx, cy, k1, k2; r,t; X)
				auto _R_sq = _Nx[0] * _Nx[0] + _Nx[1] * _Nx[1];
				auto _D = jet_t(1) + _R_sq*(_K1 + _K2 * _R_sq);
				auto _Dx = _D * _Nx[0], _Dy = _D * _Nx[1];
				auto _U = _Fx * _Dx + _Fs * _Dy + _Cx;
				auto _V = _Fy * _Dy + _Cy;

				_Residual[0] = _U0 - _U;
				_Residual[1] = _V0 - _V;

			}
			return true;
#undef   aligned_jet_t
		}

		template<typename T> __forceinline
		bool operator() (const T* const _Pars, const T* const _Pose, const T* const _Dx, const T* const _Dy, T* _Residual) const
		{
			using jet_t = T;
#define  aligned_jet_t alignas(32) jet_t
			const auto _Dot_one = jet_t(0.1);

			jet_t _Fx = m_fx * _Pars[0], _Fy = m_fy * _Pars[1], _Fs = _Pars[6];
			jet_t _Cx = m_cx * _Pars[2], _Cy = m_cy * _Pars[3];
			jet_t _K1 = _Dot_one * _Pars[4], _K2 = _Dot_one * _Pars[5];

			aligned_jet_t _X[3] = {jet_t(m_X[0]), jet_t(m_X[1]), jet_t(m_X[2])}, _Nx[3];
			_Rigid_transform(_Pose, _X, _Nx);
			_Nx[0] = _Nx[0] / _Nx[2];
			_Nx[1] = _Nx[1] / _Nx[2];

			auto _U0 = jet_t(m_pt[0] + *_Dx), _V0 = jet_t(m_pt[1] + *_Dy);
			if constexpr (_Dmodel == distortion_model::D2U) {
				auto _Du = _U0 - _Cx, _Dv = _V0 - _Cy;
				auto _Dy = _Dv / _Fy;
				auto _Dx = (_Du - _Dv * _Fs / _Fy) / _Fx;

				auto _R_sq = _Dx * _Dx + _Dy * _Dy;
				auto _D = jet_t(1) + _R_sq * (_K1 + _K2 * _R_sq);

				//C-1: x(fx, fy, fs, cx, cy, k1, k2; ¦Äu[optional]) - x'(r, t; X)
				auto _Ux = _D * _Dx, _Uy = _D * _Dy;
				_Residual[0] = _Ux - _Nx[0];
				_Residual[1] = _Uy - _Nx[1];

				//C-2: u_c(fx, fy, fs, cx, cy, k1, k2) - u'_c(fx, fy, fs; r, t; X)
				/*auto _Uu = _D * _Du, _Uv = _D * _Dv;
				auto _Ru = _Fx * _Nx[0] + _Fs * _Nx[1];
				auto _Rv = _Fy * _Nx[1];
				_Residual[0] = _Uu - _Ru;
				_Residual[1] = _Uv - _Rv;*/
			}
			if constexpr (_Dmodel == distortion_model::U2D) {
				//C: u - u'(fx, fy, fs, cx, cy, k1, k2; r,t; X)
				auto _R_sq = _Nx[0] * _Nx[0] + _Nx[1] * _Nx[1];
				auto _D = jet_t(1) + _R_sq * (_K1 + _K2 * _R_sq);
				auto _Dx = _D * _Nx[0], _Dy = _D * _Nx[1];
				auto _U = _Fx * _Dx + _Fs * _Dy + _Cx;
				auto _V = _Fy * _Dy + _Cy;

				_Residual[0] = _U0 - _U;
				_Residual[1] = _V0 - _V;

			}
			return true;
#undef   aligned_jet_t
		}

		__forceinline static
		auto Create(value_t _Fx, value_t _Fy, value_t _Cx, value_t _Cy, const point_t& _Ipt)
		{
			using cost_t = _Std_mono_error<point_t, _Dmodel>;
			return (new xop::AutoDiffCostFunction<cost_t, 2, 7, 6, 3>(
				     new cost_t(_Fx, _Fy, _Cx, _Cy, _Ipt)));
		}
		__forceinline static
		auto Create(value_t _Fx, value_t _Fy, value_t _Cx, value_t _Cy, const point_t& _Ipt, const point_t& _Ppt)
		{
			using cost_t = _Std_mono_error<point_t, _Dmodel>;
			return (new xop::AutoDiffCostFunction<cost_t, 2, 7, 6, 1, 1>(
				new cost_t(_Fx, _Fy, _Cx, _Cy, _Ipt, _Ppt)));
		}

	private:
		value_t m_fx, m_fy, m_cx, m_cy;
		point_t m_pt, m_X;
	};

	template<typename _InIt = std::add_pointer_t<value_t>> 
	struct _Lsq_stereo_errr
	{
		enum {Nres = 2, Npose = 6, Ndepth = 1};
		inline _Lsq_stereo_errr(const _InIt _Lptr, const _InIt _Rptr)
			:_Lp{ _Lptr[0], _Lptr[1] }, _Rp{ _Rptr[0], _Rptr[1] } {}

		template<typename T> bool operator() (const T* const P, const T* const w, T* residual) const
		{
			using jet_t = T;
#define  aligned_jet_t alignas(32) jet_t

			aligned_jet_t _X[] = { jet_t(_Lp[0]), jet_t(_Lp[1]), jet_t(1) };

			aligned_jet_t _x_pre[3]; _Rotate(P, _X, _x_pre);

			_x_pre[0] += P[3]**w, _x_pre[1] += P[4]**w, _x_pre[2] += P[5]**w;

			auto _x = _x_pre[0] / _x_pre[2];
			auto _y = _x_pre[1] / _x_pre[2];

			residual[0] = jet_t(_Rp[0]) - _x;
			residual[1] = jet_t(_Rp[1]) - _y;

#undef   aligned_jet_t

			return std::true_type::value;
		}

		static inline auto Create(const _InIt _Lptr, const _InIt _Rptr)
		{
			using _Error_t = _Lsq_stereo_errr<_InIt>;
			return (new xop::AutoDiffCostFunction<_Error_t, Nres, Npose, Ndepth>(new _Error_t(_Lptr, _Rptr)));
		}

	private:
		value_t _Lp[2], _Rp[2];
	};

	template<typename _Pty> struct _Std_error 
	{
		using point_t = _Pty;
		inline _Std_error(const point_t& u0, const point_t& u1)
			:m_u{ u0, u1 } {}

		template<typename T> inline
		bool operator() (const T* const K0, const T* const K1, const T* const p, const T* const w, T* residual) const
		{
			using jet_t = T;
#define  aligned_jet_t alignas(32) jet_t

			//cache the parameters
			aligned_jet_t fx[2] = { K0[0], K1[0] }, fy[2] = { K0[1], K1[1] };
			aligned_jet_t cx[2] = { K0[2], K1[2] }, cy[2] = { K0[3], K1[3] };
			aligned_jet_t k1[2] = { K0[4], K1[4] }, k2[2] = { K0[5], K1[5] };
			aligned_jet_t s[2] = { K0[6], K1[6] };

			//inverse map to normalized image domain
			aligned_jet_t u[2], v[2], x[2], y[2];
			u[0] = jet_t(m_u[0][0]), v[0] = jet_t(m_u[0][1]);
			u[1] = jet_t(m_u[1][0]), v[1] = jet_t(m_u[1][1]);
			x[0] = (u[0] - cx[0]) / fx[0], y[0] = (v[0] - cy[0]) / fy[0];
			//x[1] = (u[1] - cx[1]) / fx[1], y[1] = (v[1] - cy[1]) / fy[1];
			auto r20 = x[0] * x[0] + y[0] * y[0];
			//auto r21 = x[1] * x[1] + y[1] * y[1];
			auto und0 = jet_t(1) + r20 * (k1[0] + k2[0] * r20);
			//auto und1 = jet_t(1) + r21 * (k1[1] + k2[1] * r21);

			x[0] *= und0, y[0] *= und0;

			//back-projection and reprojection
			aligned_jet_t X[3] = { x[0], y[0], jet_t(1) }, x_pre[3];
			_Rotate(p, X, x_pre);
			x_pre[0] += *w * p[3], x_pre[1] += *w * p[4], x_pre[2] += *w * p[5];
			x[1] = x_pre[0] / x_pre[2], y[1] = x_pre[1] / x_pre[2];
			auto r21 = x[1] * x[1] + y[1] * y[1];
			auto dist = jet_t(1) + r21 * (k1[1] + k2[1] * r21);
			aligned_jet_t u_pre[2];
			u_pre[0] = x[1] * dist*fx[1] + cx[1];
			u_pre[1] = y[1] * dist*fy[1] + cy[1];
			//residual
			residual[0] = u[1] - u_pre[0];
			residual[1] = v[1] - u_pre[1];

			return true;
#undef   aligned_jet_t
		}

		static __forceinline auto Create(const point_t& u0, const point_t& u1)
		{
			enum { residuals = 2 };
			// 7 the number of internal parameters, 6 the number of structure parameters, 1 inverse depth parameter
			return new xop::AutoDiffCostFunction<
				_Std_error<point_t>, residuals, 7, 7, 6, 1>
				(new _Std_error<point_t>(u0, u1));
		}

	private:
		point_t m_u[2];
	};
	template<typename _Fty, typename _Gty> struct _Phm_error
	{
		enum { Dsize = _Fty::DescSize, Radius = _Fty::DescSize >> 1 };

		__forceinline _Phm_error(_Fty& feat, _Gty& grad) :_Feat(feat), _Grad(grad) {}
		__forceinline _Phm_error(_Fty& feat, _Gty& grad, const double* K0, const double* K1, const double* P): _Feat(feat), _Grad(grad), _K0(K0), _K1(K1), _P(P) {}

		template<typename T> inline 
		bool operator() (const T* const K0, const T* const K1, const T* const p, const T* const w, T* residuals) const
		{
			using jet_t = T;
			#define aligned_jet_t alignas(32) jet_t

			aligned_jet_t fx[2] = { K0[0], K1[0] }, fy[2] = { K0[1], K1[1] };
			aligned_jet_t cx[2] = { K0[2], K1[2] }, cy[2] = { K0[3], K1[3] };
			aligned_jet_t k1[2] = { K0[4], K1[4] }, k2[2] = { K0[5], K1[5] };
			aligned_jet_t s[2] = { K0[6], K1[6] };

			aligned_jet_t u[2], v[2], x[2], y[2];
			u[0] = jet_t(_Feat.point().x), v[0] = jet_t(_Feat.point().y);
			// \undistortion
			x[0] = (u[0] - cx[0]) / fx[0], y[0] = (v[0] - cy[0]) / fy[0];
			auto r2 = x[0] * x[0] + y[0] * y[0];
			auto _Undist = jet_t(1) + r2 * (k1[0] + k2[0] * r2);
			x[0] *= _Undist, y[0] *= _Undist;
			// \back-projection
			aligned_jet_t X[3] = { x[0], y[0], jet_t(1) };
			// \reprojection
			aligned_jet_t x_pre[3];
			_Rotate(p, X, x_pre);
			x_pre[0] += *w * p[3], x_pre[1] += *w * p[4], x_pre[2] += *w * p[5];
			x[1] = x_pre[0]/x_pre[2], y[1] = x_pre[1]/x_pre[2];
			u[1] = fx[1]*x[1]+cx[1], v[1] = fy[1] * y[1] + cy[1];
			// \similarity optimization
			const auto& _Dp = _Feat.desc();
			const auto& _Im = _Grad.img();
			const auto& _Gx = _Grad.get<0>();
			const auto& _Gy = _Grad.get<1>();
			for (auto y = -Radius, j = 0; y <= Radius; ++y, ++j) {
				const auto _V = v[1] + jet_t(y);
				for (auto x = -Radius, i = 0; x <= Radius; ++x, ++i) {
					const auto _U = u[1] + jet_t(x);
					const auto _I0 = jet_t(_Dp(i, j));
					const auto _I1 = _Interp_impl(_Im, _Gx, _Gy, _U, _V);
					residuals[j*Dsize + i] = (_I0 - _I1);
				}
			}
			residuals[Dsize*Dsize] = u[0] - u[1];
			residuals[Dsize*Dsize+1] = v[0] - v[1];
			return true;
			#undef aligned_jet_t
		}

		template<typename T> inline bool operator()(const T* const w, T* residuals) const
		{
			using jet_t = T;
#define aligned_jet_t alignas(32) jet_t

			aligned_jet_t fx[2] = { _K0[0], _K1[0] }, fy[2] = { _K0[1], _K1[1] };
			aligned_jet_t cx[2] = { _K0[2], _K1[2] }, cy[2] = { _K0[3], _K1[3] };
			aligned_jet_t k1[2] = { _K0[4], _K1[4] }, k2[2] = { _K0[5], _K1[5] };
			aligned_jet_t s[2] = { _K0[6], _K1[6] };

			aligned_jet_t u[2], v[2], x[2], y[2];
			u[0] = jet_t(_Feat.point().x), v[0] = jet_t(_Feat.point().y);
			// \undistortion
			x[0] = (u[0] - cx[0]) / fx[0], y[0] = (v[0] - cy[0]) / fy[0];
			auto r2 = x[0] * x[0] + y[0] * y[0];
			auto _Undist = jet_t(1) + r2 * (k1[0]/* + k2[0] * r2*/);
			x[0] *= _Undist, y[0] *= _Undist;
			// \back-projection
			aligned_jet_t X[3] = { x[0], y[0], jet_t(1) };
			// \reprojection
			aligned_jet_t x_pre[3];
			_Rotate(_P, X, x_pre);
			x_pre[0] += *w * _P[3], x_pre[1] += *w * _P[4], x_pre[2] += *w * _P[5];
			x[1] = x_pre[0] / x_pre[2], y[1] = x_pre[1] / x_pre[2];
			auto _Dist1 = jet_t(1) + k1[1]*(x[1] * x[1] + y[1] * y[1]);
			auto x_dist = x[1] * _Dist1, y_dist = y[1] * _Dist1;
			u[1] = fx[1] * x_dist + cx[1], v[1] = fy[1] * y_dist + cy[1];
			// \similarity optimization
			for (auto y = -Radius, j = 0; y <= Radius; ++y, ++j) {
				const auto _V = v[1] + jet_t(y);
				for (auto x = -Radius, i = 0; x <= Radius; ++x, ++i) {
					const auto _U = u[1] + jet_t(x);
					const auto _I0 = jet_t(_Dp(i, j));
					const auto _I1 = _Interp_impl(_Im, _Gx, _Gy, _U, _V);
					residuals[j*Dsize + i] = (_I0 - _I1);
				}
			}
			return true;
#undef aligned_jet_t
		}

		static inline xop::CostFunction* Create(_Fty& feat, _Gty& grad) {
			enum { residuals = Dsize * Dsize + 2 };
			return new xop::AutoDiffCostFunction<
				// 7 the number of internal parameters, 6 the number of structure parameters, 1 inverse depth parameter
				_Phm_error<_Fty, _Gty>, residuals, 7, 7, 6, 1>
				(new _Phm_error<_Fty, _Gty>(feat, grad));
		}
		template<typename... _Args>
		static inline xop::CostFunction* Create(_Fty& feat, _Gty& grad, _Args... _args) {
			enum { residuals = Dsize * Dsize + 2 };
			return new xop::AutoDiffCostFunction<
				// 1 inverse depth parameter
				_Phm_error<_Fty, _Gty>, residuals, 1>
				(new _Phm_error<_Fty, _Gty>(feat, grad, _args...));
		}

		_Fty& _Feat;
		_Gty& _Grad;
		const detail::Matrix_<typename _Fty::value_t, Dsize, Dsize>& _Dp = _Feat.desc();
		const typename _Gty::Image_t& _Im = _Grad.img();
		const typename _Gty::Matrx_t& _Gx = _Grad.get<0>();
		const typename _Gty::Matrx_t& _Gy = _Grad.get<1>();
		double* _K0, _K1, _P;
	};
};
}
