/*****************************************************************
 This file is part of monocular camera calibration library.
 Copyright(C) 2018-2020, Zhilong (Dgelom) Su, all rights reserved.
******************************************************************/
#pragma once
#include <Matrice/io/io.hpp>
#include <Matrice/forward.hpp>

#define DGE_HOST_INLINE __forceinline

namespace dgelom {
///<\brief> Lens distortion model <\brief>
enum class distortion_model
{
	U2D = 0x0002,
	D2U = 0x0003,
};

///<\brief> calibration target pattern <\brief>
enum class pattern_type
{
	squared = 0,
	circular = 1,
};

template<pattern_type _Type = pattern_type::squared>
struct pattern
{
	static constexpr auto type = _Type;

	DGE_HOST_INLINE pattern() noexcept {
	}
	DGE_HOST_INLINE pattern(size_t rows, size_t cols, float spacing) noexcept
		: m_rows(rows), m_cols(cols), m_spacing(spacing) {
	}

	DGE_HOST_INLINE bool empty() const noexcept {
		return m_rows == 0 || m_cols == 0 || m_spacing == 0;
	}
	DGE_HOST_INLINE size_t count() const noexcept {
		return m_rows * m_cols;
	}
	template<typename size_type>
	DGE_HOST_INLINE size_type size() const noexcept {
		return size_type(m_cols, m_rows);
	}
	DGE_HOST_INLINE size_t pitch() const noexcept {
		return m_spacing;
	}
	DGE_HOST_INLINE size_t rows() const noexcept {
		return m_rows;
	}
	DGE_HOST_INLINE size_t cols() const noexcept {
		return m_cols;
	}

private:
	size_t m_rows, m_cols;
	float m_spacing;
};
namespace detail {
template<typename _Ty, pattern_type _Pt> 
class _Calibrator {
	static constexpr auto Npars = 7, Npose = 6;
	static constexpr auto Dmodel = distortion_model::U2D;
public:
	using value_t = _Ty;
	using image_info_t = IO::Dir_<0>;
	using plane_array = Matrix_<value_t, Npars, 1>;
	using matrix_t = Matrix<value_t>;
	using ptarray_t = Matrix<float>;
	using matrix3x3 = Matrix_<value_t, 3, 3>;
	using pattern_t = pattern<_Pt>;

	_Calibrator(const pattern_t& _Pattern);
	_Calibrator(const image_info_t& _Fnames, const pattern_t& _Pattern);

	// \perform calibration and return internal params with [fx, fy, cx, cy, k1, k2, ks]
	plane_array& run(bool require_optim = true);

	// \get external parameters with [r, t \\ ... \\ r, t]
	const matrix_t& get_poses() const noexcept;

	// \get planar model points
	const ptarray_t& planar_points() const noexcept;
	ptarray_t& planar_points() noexcept;

	// \get all image points
	const ptarray_t& image_points() const noexcept;
	ptarray_t& image_points() noexcept;

	// \get valid image indices
	DGE_HOST_INLINE decltype(auto) valid_image_indices() const {
		return (m_effindices);
	}

	// \image width and height
	DGE_HOST_INLINE size_t& image_width() {
		return m_iw;
	}
	DGE_HOST_INLINE size_t& image_height() {
		return m_ih;
	}
	DGE_HOST_INLINE const size_t& image_width() const {
		return m_iw;
	}
	DGE_HOST_INLINE const size_t& image_height() const {
		return m_ih;
	}

	DGE_HOST_INLINE value_t& scale() {
		return m_scale;
	};
	DGE_HOST_INLINE const value_t& scale() const {
		return m_scale;
	};
	DGE_HOST_INLINE value_t& error() {
		return m_error;
	};
	DGE_HOST_INLINE const value_t& error() const {
		return m_error;
	};

private:
	void _Get_image_points();
	void _Get_planar_points();
	void _Get_true_to_scale();
	ptarray_t _Normalize(size_t i);
	matrix3x3 _Find_homography(size_t i);
	plane_array& _Analysis();
	void _Update_dist_eqs(size_t i, Matrix_<value_t, ::dynamic, 2>& A, Matrix_<value_t, ::dynamic, 1>& b, const matrix3x3& R, const value_t* T);
	void _Optimize();

	plane_array m_params = 0; //fx, fx, cx, cy, k1, k2, fs
	matrix_t m_poses;
	value_t m_error;
	value_t m_scale = matrix_t::inf;

	pattern_t m_pattern;
	image_info_t m_fnames;
	size_t m_iw, m_ih;

	ptarray_t m_points;
	ptarray_t m_imgpts;
	std::vector<size_t> m_effindices;
	matrix3x3 m_normal = matrix3x3::inf;
};
}
using chessboard_calibration = detail::_Calibrator<double_t, pattern_type::squared>;
using circular_calibration = detail::_Calibrator<double_t, pattern_type::circular>;
}

#undef DGE_HOST_INLINE