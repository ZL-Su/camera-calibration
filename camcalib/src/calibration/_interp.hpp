#pragma once
#include <tuple>
#include "_jet_extras.hpp"

namespace dgelom {
template<typename _Valty> inline
_Valty _Tidy_axis(_Valty x, int limx, int& x0, int& x1)
{
	const auto ix = static_cast<int>(x);
	if (ix < 0) { x0 = x1 = 0; return _Valty(0); }
	if (ix > limx - 2) { x0 = x1 = limx - 1; return _Valty(1); }
	x0 = ix, x1 = ix + 1; return _Valty(x1 - x);
}
template<typename _Imty, typename _Gdty, typename _Valty>
_Valty _Bilinear_interp(const _Imty& img, const _Gdty& gx, const _Gdty& gy, _Valty x[2])
{
	using Image_t = _Imty;
	using Matrx_t = _Gdty;
	using pixel_t = typename Image_t::value_t;
	using value_t = std::common_type_t<typename Matrx_t::value_t, _Valty>;

	int x0, x1, y0, y1;
	auto dx = _Tidy_axis(x[0], img.cols(), x0, x1);
	auto dy = _Tidy_axis(x[1], img.rows(), y0, y1);
	auto dxt = 1.0 - dx, dyt = 1.0 - dy;
	auto _Interp = [&](auto _F11, auto _F12, auto _F21, auto _F22)->auto 
	     { return dy * (dx*_F11 + dxt*_F12) + dyt*(dx*_F21 + dxt*_F22); };

	auto Gx11 = value_t(gx(x0, y0)), Gx12 = value_t(gx(x1, y0));
	auto Gx21 = value_t(gx(x0, y1)), Gx22 = value_t(gx(x1, y1));
	x[0] = _Interp(Gx11, Gx12, Gx21, Gx22);

	auto Gy11 = value_t(gy(x0, y0)), Gy12 = value_t(gy(x1, y0));
	auto Gy21 = value_t(gy(x0, y1)), Gy22 = value_t(gy(x1, y1));
	x[1] = _Interp(Gy11, Gy12, Gy21, Gy22);

	auto I11 = value_t(img(x0, y0)), I12 = value_t(img(x1, y0));
	auto I21 = value_t(img(x0, y1)), I22 = value_t(img(x1, y1));
	return _Interp(I11, I12, I21, I22);
}
template<typename T, typename _Imty, typename _Gdty>
T _Interp_impl(const _Imty& img, const _Gdty& gx, const _Gdty& gy, const T& x, const T& y)
{
	using Image_t = _Imty;
	using Matrx_t = _Gdty;
	using pixel_t = typename Image_t::value_t;
	using value_t = typename Matrx_t::value_t;

	auto scalar_x = ceres::JetOps<T>::GetScalar(x);
	auto scalar_y = ceres::JetOps<T>::GetScalar(y);

	value_t _Grad[2] = { scalar_x, scalar_y };
	auto _Inte = _Bilinear_interp(img, gx, gy, _Grad);

	T _X[2] = { x, y };
	return ceres::Chain<value_t, 2, T>::Rule(_Inte, _Grad, _X);
}
}
