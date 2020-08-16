/**************************************************************************
This file is part of Matrice, an effcient and elegant C++ library.
Copyright(C) 2018, Zhilong(Dgelom) Su, all rights reserved.

This program is free software : you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.If not, see <http://www.gnu.org/licenses/>.
**************************************************************************/
#pragma once

#include "..\core\storage.hpp"
#include "..\core\_expr_type_traits.h"
#include "..\core\matrix_expr.hpp"
#include "..\..\..\addin\interface.h"
#ifdef __enable_blak__
#include <lapacke.h>
#endif // __enable_blak__

#pragma warning(disable: 4715 4661 4224 4267 4244)

namespace Dgelo {
#ifndef MATRICE_INT
#define MATRICE_INT                                std::ptrdiff_t
	typedef MATRICE_INT                                     int_t;
#else
	typedef int                                             int_t;
#endif // !MATRICE_INT

namespace types {

template<typename _Ty> using nested_initializer_list = std::initializer_list<std::initializer_list<_Ty>>;

template<typename _Ty, int _M, int _N> class Matrix_;
template<typename _Ty, int _M, int _N, typename _Derived> class Base_;
template<typename _Ty, int _M, int _N>
using EwiseMulExprMM = exprs::Expr::EwiseBinaryExpr<Base_<_Ty, _M, _N, Matrix_<_Ty, _M, _N>>, Base_<_Ty, _M, _N, Matrix_<_Ty, _M, _N>>, std::multiplies<_Ty>>;

/*******************************************************************
	              Generic Base for Matrix Class
	          Copyright (c) : Zhilong Su 14/Feb/2018
 ******************************************************************/
template<typename _Ty, int _M, int _N, typename _Derived> class Base_ : public
#ifdef __use_ocv_as_view__
	ocv_view_t
#endif
{
	typedef details::Storage_<_Ty>::Allocator<_M, _N, -1>         Storage;
	typedef Base_                                                    _Myt;
	typedef _Myt&                                                 myt_ref;
	typedef const _Myt&                                     const_myt_ref;
	typedef _Myt&&                                               myt_move;
public:
	typedef typename details::Storage_<_Ty>::value_t              value_t;
	typedef typename details::Storage_<_Ty>::pointer              pointer;
	typedef typename details::Storage_<_Ty>::reference          reference;
	typedef typename details::Storage_<_Ty>::idx_t                  idx_t;
	typedef typename details::Location                            loctn_t;
	typedef enum StorageFormat { RowMajor, ColMajor }            format_t;
	constexpr static const _Ty inf = std::numeric_limits<_Ty>::infinity();
	constexpr static const _Ty eps = std::numeric_limits<_Ty>:: epsilon();
	MATRICE_GLOBAL_INL Base_() noexcept 
		: m_storage(), m_data(m_storage.data()), m_rows(_M < 0 ? 0:_M), m_cols(_N < 0 ? 0 : _N) { _Init_view(); }
	MATRICE_GLOBAL_INL Base_(int _rows, int _cols) noexcept
		: m_storage(_rows, _cols), m_data(m_storage.data()), m_rows(_rows), m_cols(_cols) { _Init_view(); }
	MATRICE_GLOBAL_INL Base_(int _rows, int _cols, pointer data) noexcept
		: m_storage(_rows, _cols, data), m_data(m_storage.data()), m_rows(_rows), m_cols(_cols) { _Init_view(); }
	MATRICE_GLOBAL_INL Base_(int _rows, int _cols, const value_t _val) noexcept
		: m_storage(_rows, _cols, _val), m_data(m_storage.data()), m_rows(_rows), m_cols(_cols) { _Init_view(); }
	MATRICE_GLOBAL_INL Base_(const std::initializer_list<value_t> _list)
		:m_storage(_list), m_data(m_storage.data()), m_rows(_M), m_cols(_N) { _Init_view(); }
	MATRICE_GLOBAL_INL Base_(const_myt_ref _other) noexcept
		: m_storage(_other.m_storage), m_data(m_storage.data()), m_rows(_other.rows()), m_cols(_other.cols()) { _Init_view(); }
	MATRICE_GLOBAL_INL Base_(myt_move _other) noexcept
		: m_rows(_other.rows()), m_cols(_other.cols()), m_storage(std::move(_other.m_storage)), m_data(m_storage.data()) { _Init_view(); }
public:
	///<brief> dynamic create methods </brief>
	MATRICE_HOST_ONLY void create(int_t rows, int_t cols) { if(_M == 0) static_cast<_Derived*>(this)->create(rows, cols); };
	///<brief> accessors </brief>
	MATRICE_GLOBAL_INL pointer   operator[](int_t y) const { return (m_data + y * m_cols); }
	MATRICE_GLOBAL_INL reference operator() (int_t i) const { return m_data[i]; }
	MATRICE_GLOBAL_INL reference operator() (int_t i) { return m_data[i]; }
	template<format_t _Fmt = RowMajor>
	MATRICE_GLOBAL_INL reference operator() (int _r, int _c) const { return (m_data + (_Fmt == RowMajor ? m_cols : m_rows) * _r)[_c]; }
	///<brief> access methods </brief>
	MATRICE_GLOBAL_FINL pointer data() const { return (m_data); }
	MATRICE_GLOBAL_FINL pointer ptr(int y = 0) const { return (m_data + m_cols * y); }
	MATRICE_GLOBAL_FINL int_t rows() const { return m_rows; }
	MATRICE_GLOBAL_FINL int_t cols() const { return m_cols; }
	MATRICE_GLOBAL_FINL std::size_t size() const { return static_cast<std::size_t>(m_storage.size()); }
	MATRICE_GLOBAL_INL _Myt view(const nested_initializer_list<std::size_t> _plain_list) const;
	///<brief> assignment operators </brief>
	MATRICE_GLOBAL_INL _Myt& operator= (const _Myt& _other) //homotype assignment operator
	{
		m_storage = _other.m_storage;// m_data = m_storage.data();
		m_cols = _other.m_cols, m_rows = _other.m_rows;
		m_format = _other.m_format;
		return (*this);
	}
	template<int _Rows, int _Cols> //static to dynamic
	MATRICE_GLOBAL_INL _Derived& operator= (Matrix_<value_t, _Rows, _Cols>& _managed)
	{
		m_data = _managed.data();
		m_rows = _Rows, m_cols = _Cols;
		m_storage.owner() = details::Storage_<value_t>::Proxy;
		_Init_view();
		return (*static_cast<_Derived*>(this));
	}

#pragma region ///<brief> Lazied Operations for Matrix Arithmetic </brief>
	template<typename _RhsType>
	MATRICE_GLOBAL_INL exprs::Expr::EwiseBinaryExpr<_Myt, _RhsType, std::plus<value_t>> operator+ (const _RhsType& _opd) const
	{
		return exprs::Expr::EwiseBinaryExpr<_Myt, _RhsType, std::plus<value_t>>(*this, _opd);
	}
	template<typename _RhsType>
	MATRICE_GLOBAL_INL exprs::Expr::EwiseBinaryExpr<_Myt, _RhsType, std::minus<value_t>> operator- (const _RhsType& _opd) const
	{
		return exprs::Expr::EwiseBinaryExpr<_Myt, _RhsType, std::minus<value_t>>(*this, _opd);
	}
	template<typename _RhsType>
	MATRICE_GLOBAL_INL exprs::Expr::EwiseBinaryExpr<_Myt, _RhsType, std::multiplies<value_t>> operator* (const _RhsType& _opd) const
	{
		return exprs::Expr::EwiseBinaryExpr<_Myt, _RhsType, std::multiplies<value_t>>(*this, _opd);
	}
	template<typename _RhsType>
	MATRICE_GLOBAL_INL exprs::Expr::EwiseBinaryExpr<_Myt, _RhsType, std::divides<value_t>> operator/ (const _RhsType& _opd) const
	{
		return exprs::Expr::EwiseBinaryExpr<_Myt, _RhsType, std::divides<value_t>>(*this, _opd);
	}
	template<typename _RhsType>
	MATRICE_GLOBAL_INL exprs::Expr::MatBinaryExpr<_Myt, _RhsType, exprs::Expr::Op::MatMul<value_t>> mul(const _RhsType& rhs) //const
	{
		return exprs::Expr::MatBinaryExpr<_Myt, _RhsType, exprs::Expr::Op::MatMul<value_t>>(*this, rhs);
	}
	MATRICE_HOST_FINL exprs::Expr::MatUnaryExpr<_Myt, exprs::Expr::Op::Inverse<value_t>> inv()
	{
		return exprs::Expr::MatUnaryExpr<_Myt, exprs::Expr::Op::Inverse<value_t>>(*this);
	}
	MATRICE_GLOBAL_INL exprs::Expr::EwiseBinaryExpr<_Myt, _Myt, std::multiplies<value_t>> normalize(value_t _val = inf)
	{
		return ((std::abs(_val) < eps ? value_t(1) : value_t(1) / (_val == inf ? max() : _val))*(*this));
	}
#pragma endregion

#pragma region ///<brief> Triggers for Suspended Expression </brief>
	template<typename _Opd1, typename _Opd2, typename _ExprOp>
	MATRICE_GLOBAL_INL _Derived& operator= (const exprs::Expr::EwiseBinaryExpr<_Opd1, _Opd2, _ExprOp>& expr)
	{
		int_t N = std::min(this->size(), expr.size());
#ifdef __enable_omp__
#pragma omp parallel if(N > 100)
		{
#pragma omp for nowait
#endif
			for (int_t i = 0; i < N; ++i) m_data[i] = expr(i);
#ifdef __enable_omp__
		}
#endif
		return (*static_cast<_Derived*>(this));
	}
	template<typename _Opd1, typename _Opd2, typename _ExprOp>
	MATRICE_GLOBAL_INL _Derived& operator= (const exprs::Expr::MatBinaryExpr<_Opd1, _Opd2, _ExprOp>& expr)
	{
		for (int r = 0; r < m_rows; ++r) for (int c = 0; c < m_cols; ++c)
			this->operator[](r)[c] = expr(r, c);
		return (*static_cast<_Derived*>(this));
	}
	MATRICE_GLOBAL_INL _Derived& operator= (const exprs::Expr::MatUnaryExpr<_Myt, exprs::Expr::Op::Inverse<value_t>>& _expr)
	{
		this->m_data = _expr(); return (*static_cast<_Derived*>(this));
	}
#pragma endregion

	///<brief> in-time matrix arithmetic </brief>
	MATRICE_GLOBAL_INL value_t max() const { return (*std::max_element(m_data, m_data + size())); }
	MATRICE_GLOBAL_INL value_t min() const { return (*std::min_element(m_data, m_data + size())); }
	MATRICE_HOST_ONLY  value_t det() const
	{
		if 
#ifdef _HAS_CXX17
			//constexpr 
#endif
			(type_bytes<value_t>::value == 8) return fblas::_ddetm(fkl::dptr(m_data), m_rows);
		else return std::numeric_limits<value_t>::infinity();
	}
	template<typename _RhsType>
	MATRICE_GLOBAL_INL value_t dot(const _RhsType& _rhs) const
	{
		return (this->operator*(_rhs)).sum();
	}
	MATRICE_GLOBAL_INL value_t norm_2() const
	{
		auto ans = dot(*this); return (ans > eps ? std::sqrt(ans) : inf);
	}
#ifdef __use_ocv_as_view__
	operator ocv_view_t() { return *static_cast<ocv_view_t*>(this); }
#endif

	///<brief> properties </brief>
	__declspec(property(get = _Prop_format_getter)) format_t format;
	MATRICE_HOST_INL format_t _Prop_format_getter() { return m_format; }
	__declspec(property(get = _Prop_empty_getter)) bool empty;
	MATRICE_HOST_INL bool _Prop_empty_getter() const { return (size() == 0); }

protected:
	MATRICE_HOST_INL void _Init_view()
	{
#ifdef __use_ocv_as_view__
		*static_cast<ocv_view_t*>(this) = ocv_view_t(m_rows, m_cols, ocv_view_t_cast<value_t>::type, m_data);
#endif // __use_ocv_as_view__
	}

protected:
	int m_rows, m_cols;
	Storage m_storage;
	format_t m_format = RowMajor;
	pointer m_data = nullptr;
};
template<typename _RhsType, typename = std::enable_if<std::is_arithmetic<typename _RhsType::value_t>::value, typename _RhsType::value_t>::type>
MATRICE_GLOBAL_INL exprs::Expr::EwiseBinaryExpr<_RhsType, _RhsType, std::multiplies<typename _RhsType::value_t>> operator* (const typename _RhsType::value_t& _scalar, const _RhsType& _opd)
{
	return exprs::Expr::EwiseBinaryExpr<_RhsType, _RhsType, std::multiplies<typename _RhsType::value_t>>(_scalar, _opd);
}
template<typename _LhsType, typename = std::enable_if<std::is_arithmetic<typename _LhsType::value_t>::value, typename _LhsType::value_t>::type>
MATRICE_GLOBAL_INL exprs::Expr::EwiseBinaryExpr<_LhsType, _LhsType, std::multiplies<typename _LhsType::value_t>> operator* (const _LhsType& _opd, const typename _LhsType::value_t& _scalar)
{
	return exprs::Expr::EwiseBinaryExpr<_LhsType, _LhsType, std::multiplies<typename _LhsType::value_t>>(_scalar, _opd);
}
template<typename _LhsType, typename _RhsType>
MATRICE_GLOBAL_INL exprs::Expr::EwiseBinaryExpr<_LhsType, _RhsType, std::multiplies<typename _LhsType::value_t>> operator* (const _LhsType& _left, const _RhsType& _right)
{
	return exprs::Expr::EwiseBinaryExpr<_LhsType, _RhsType, std::multiplies<typename _LhsType::value_t>>(_left, _right);
}

/*******************************************************************
	Generic Matrix Class with Aligned Static Memory Allocation
	          Copyright (c) : Zhilong Su 14/Feb/2018
*******************************************************************/
template<typename _Ty, int _M, int _N>
class Matrix_ : public Base_<_Ty, _M, _N, Matrix_<_Ty, _M, _N>>
{
	typedef Base_<_Ty, _M, _N, Matrix_<_Ty, _M, _N>> base_t;
	typedef Matrix_                                     Myt;
	typedef const Matrix_&                     const_my_ref;
	using base_t::m_rows;
	using base_t::m_cols;
	using base_t::m_data;
public:
	using typename base_t::value_t;
	using typename base_t::pointer;
	MATRICE_GLOBAL_INL Matrix_(int _pld1 = 0, int _pld2 = 0) noexcept : base_t() {};
	MATRICE_GLOBAL_INL Matrix_(pointer data) noexcept : base_t(_M, _N, data) {};
	MATRICE_GLOBAL_INL Matrix_(const_my_ref _other) noexcept : base_t(_other) {};
	MATRICE_HOST_INL Matrix_(const std::initializer_list<value_t> _list) noexcept : base_t(_list) {}

	MATRICE_GLOBAL_INL operator Matrix_<value_t, 0, 0>() const 
	{ return Matrix_<value_t, 0, 0>(m_rows, m_cols, m_data); }

	template<typename _Expr>
	MATRICE_GLOBAL_INL Myt& operator= (const _Expr& _expr) { return base_t::operator=(_expr); }
};

/*******************************************************************
    Generic Matrix Class with Aligned Dynamic Memory Allocation
	         Copyright (c) : Zhilong Su 14/Feb/2018
 ******************************************************************/
template<typename _Ty>
class Matrix_<_Ty, compile_time_size<>::RunTimeDeduceInHost, compile_time_size<>::RunTimeDeduceInHost> : public Base_<_Ty, 0, 0, Matrix_<_Ty, 0, 0>>
{
	typedef Base_<_Ty, 0, 0, Matrix_<_Ty, 0, 0>> base_t;
	typedef typename base_t::pointer            pointer;
	typedef Matrix_                                 Myt;
	typedef const Matrix_&                 const_my_ref;
	using base_t::m_data;
	using base_t::m_rows;
	using base_t::m_cols;
	using base_t::m_storage;
public:
	typedef typename base_t::value_t value_t;
	MATRICE_GLOBAL_INL Matrix_() noexcept : base_t() {};
	MATRICE_GLOBAL_INL Matrix_(int _rows) noexcept : base_t(_rows, 1) {};
	MATRICE_GLOBAL_INL Matrix_(int _rows, int _cols) noexcept : base_t(_rows, _cols) {};
	MATRICE_GLOBAL_INL Matrix_(int _rows, int _cols, pointer _data) noexcept : base_t(_rows, _cols, _data) {};
	MATRICE_GLOBAL_INL Matrix_(int _rows, int _cols, const value_t _val) noexcept : base_t(_rows, _cols, _val) {};
	MATRICE_GLOBAL_INL Matrix_(const_my_ref _other) noexcept : base_t(_other) {};
	template<int _M, int _N>
	MATRICE_GLOBAL_INL Matrix_(Matrix_<_Ty, _M, _N>& _managed) noexcept : base_t(_M, _N, _managed.data()) {};

	MATRICE_GLOBAL void create(int_t rows, int_t cols = 1);

	template<typename _Expr>
	MATRICE_GLOBAL_INL Myt& operator= (const _Expr& _expr) { return base_t::operator=(_expr); }

};

template<typename _Ty> 
using Matrix = Matrix_<_Ty, compile_time_size<>::RunTimeDeduceInHost, compile_time_size<>::RunTimeDeduceInHost>;
template<int _Rows, int _Cols>
using Matrixf = Matrix_<float, compile_time_size<_Rows, _Cols>::CompileTimeRows, compile_time_size<_Rows, _Cols>::CompileTimeCols>;
template<int _Rows, int _Cols>
using Matrixd = Matrix_<double, compile_time_size<_Rows, _Cols>::CompileTimeRows, compile_time_size<_Rows, _Cols>::CompileTimeCols>;

#pragma region Matrix base which using storage class as template arg. 
template<typename _Ty, class _Storage = details::Storage_<_Ty>::
#ifdef __CXX11__
	SharedAllocator<_Ty>
#else
	DynamicAllocator<_Ty>
#endif
>
class MatrixBase_ : public
#ifdef __use_ocv_as_view__
	ocv_view_t
#endif
{
	typedef typename details::Storage_<_Ty>::value_t   value_t;
	typedef typename details::Storage_<_Ty>::pointer   pointer;
	typedef typename details::Storage_<_Ty>::reference reference;
	typedef typename details::Storage_<_Ty>::idx_t     idx_t;
	typedef typename details::Location                 loctn_t;
	typedef MatrixBase_ Myt;
public:
	explicit MatrixBase_(const int_t _m, const int_t _n) noexcept;
	explicit MatrixBase_() noexcept 
		: m_storage(), m_rows(m_storage.rows()), m_cols(m_storage.cols())
	{
		if (m_storage.location() == loctn_t::OnStack)
		{
			m_data = m_storage.data();
#ifdef __use_ocv_as_view__
			*static_cast<ocv_view_t*>(this) = ocv_view_t(m_storage.rows(), m_storage.cols(), ocv_view_t_cast<value_t>::type, m_data);
#endif // __use_ocv_as_view__
		}
		//else throw("This constructor is only for managed allocator.");
	};

public:
	inline value_t& operator[] (int_t idx) const { return m_data[idx]; }
	inline value_t& operator() (int_t ridx, int_t cidx) const { return m_data[cidx+ridx* m_cols]; }
	inline pointer ptr(int_t ridx) const { return (m_data + ridx*m_cols); }
	inline value_t& colmaj(int_t ridx, int_t cidx) const { return m_data[ridx + cidx * m_rows]; }

protected:
	pointer m_data;
	_Storage m_storage;
private:
	std::size_t m_rows, m_cols;
};
/*template<typename _Ty, int N1, int N2>
class MatrixN_ : public MatrixBase_<_Ty, details::Storage_<_Ty>::ManagedAllocator<N1,N2>>
{
	typedef MatrixBase_<_Ty, details::Storage_<_Ty>::ManagedAllocator<N1,N1>> Base_t;
public:
	MatrixN_() noexcept : Base_t() {};
};
template<typename _Ty>
class MatrixD_ : public MatrixBase_<_Ty>
{
	typedef MatrixBase_<_Ty> Base_t;
public:
	MatrixD_() noexcept : Base_t() {};
	MatrixD_(int_t m, int_t n) noexcept : Base_t(m, n) {};
};*/
#pragma endregion
}
}
