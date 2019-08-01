/***************************************************************************
This file is part of monocular camera calibration library.
Copyright(C) 2018, Zhilong (Dgelom) Su, all rights reserved.

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
#include <Matrice\io\io.h>
#include <Matrice\core\matrix.h>

namespace dgelom {
#define INLINE MATRICE_HOST_FINL
enum {analytic = 0x0001, optimized = 0x0000};
enum { D2U = 0x0003, U2D = 0x0002 };
enum pattern_type {squared = 0, circular = 1, };
template<pattern_type _Type = squared> struct pattern
{
	enum { type = _Type };
	INLINE pattern() {}
	INLINE pattern(size_t rows, size_t cols, float size)
		: m_rows(rows), m_cols(cols), m_size(size) {}

	INLINE bool empty() { return m_rows == 0 || m_cols == 0 || m_size == 0; }
	INLINE size_t count() const { return m_rows * m_cols; }
	template<typename _Sizety>
	INLINE _Sizety size() const { return _Sizety(m_cols, m_rows); }
	INLINE size_t pitch() const { return m_size; }
	INLINE size_t rows() const { return m_rows; }
	INLINE size_t cols() const { return m_cols; }
private:
	size_t m_rows, m_cols; float m_size;
};
template<typename T, pattern_type _Patt = squared, size_t _Order = 2> 
class mono_calibrator
{
	enum { Npars = 2 + 2 + _Order + 1, Npose = 6, Dmodel = U2D };
	using img_info_t = IO::Dir_<0>;
	std::vector<std::future<void>> _My_futures;
	std::future<void> _My_future_scale;
public:
	using plane_array = types::Matrix_<T, Npars, compile_time_size<>::val_1>;
	using value_t = typename plane_array::value_t;
	using matrix_t = types::Matrix<T>;
	using ptarray_t = types::Matrix<float>;
	using matrix3x3 = types::Matrix_<value_t, 3, 3>;
	using pattern_t = pattern<_Patt>;

	// \constructor for image points fed by user
	INLINE mono_calibrator(const pattern_t& _Pattern)
		:m_pattern(_Pattern) {
		_Get_planar_points();
	}
	// \constructor for image path fed by user
	INLINE mono_calibrator(const img_info_t& _Fnames, const pattern_t& _Pattern) 
		:m_fnames(_Fnames), m_pattern(_Pattern) {
		_Get_planar_points();
		_Get_image_points();
	}

	// \get planar model points
	INLINE decltype(auto) planar_points() { return _Retrieve_from_bg(); }
	INLINE const decltype(auto) planar_points() const { return _Retrieve_from_bg(); }

	// \get i-th image points
	INLINE ptarray_t& image_points(size_t i) { return (m_ipoints[i]); }
	INLINE const ptarray_t& image_points(size_t i) const { return (m_ipoints[i]); }
	INLINE const ptarray_t& image_indices() const { return (m_indices);}

	// \image width and height
	INLINE size_t& image_width() { return m_iw; }
	INLINE size_t& image_height() { return m_ih; }
	INLINE const size_t& image_width() const { return m_iw; }
	INLINE const size_t& image_height() const { return m_ih; }

	INLINE value_t& scale() { return m_scale; };
	INLINE const value_t& scale() const { return m_scale; };
	INLINE value_t& error() { return m_error; };
	INLINE const value_t& error() const { return m_error; };
 
	// \perform calibration and return internal params
	template<size_t _Option = optimized, size_t _Dmodel = D2U>
	INLINE plane_array& run();
	
	INLINE plane_array operator()() const { return (m_params);}
	INLINE decltype(auto) operator()(size_t i) const {
		return types::Vec_<value_t, 6>{m_poses[i][0], m_poses[i][1], m_poses[i][2], m_poses[i][3], m_poses[i][4], m_poses[i][5]};
	}
private:
	INLINE auto& _Get_image_points();
	INLINE auto& _Get_planar_points();
	INLINE void _Retrieve_from_bg();
	INLINE void _Normalize_points();
	INLINE void _Get_true_to_scale();
	INLINE matrix3x3 _Find_homography(size_t i);
	INLINE plane_array& _Analysis() noexcept;
	template<size_t _Dmodel>
	INLINE void _Optimize(plane_array& _Params);

	plane_array m_params = { 0 }; //fx, fx, cx, cy, k1, k2, fs
	matrix_t m_poses;
	value_t m_error;
	value_t m_scale = matrix_t::inf;

	pattern_t m_pattern;
	img_info_t m_fnames;
	size_t m_iw, m_ih;

	ptarray_t m_points;
	std::vector<ptarray_t> m_ipoints;
	std::vector<size_t> m_indices;
	matrix3x3 m_normal = { matrix3x3::inf };
};
}
#include "inl\_mono_impl.inl"
