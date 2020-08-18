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
//#include "matrix.h"
#include <functional>
#include <cassert>
#ifdef __AVX__
#include <immintrin.h>
#endif
#include "_expr_type_traits.h"
#include "../util/_macros.h"
#include "../core/storage.hpp"

#pragma warning(disable: 4715)

MATRICE_NAMESPACE_EXPR_BEGIN

template<typename _Ty> class Matrix;
template<typename _Ty, int _Rows, int _cols> class Matrix_;
template<class _T1, class _T2, typename _BinaryOp> class MatBinaryExpr;
template<class _Oprd, typename _UnaryOp> class MatUnaryExpr;
struct Op { template<typename _Ty> struct MatMul; };
struct Expr {
	/*\breif: factorial_t<N> = N!*/
	template<int N> struct factorial_t { 
		enum { value = factorial_t<N - 1>::value*N }; 
	};
	template<> struct factorial_t<0> { 
		enum { value = 1 }; 
	};

	struct Op {
	template<typename _Ty> using EwiseSum = std::plus<_Ty>;
	template<typename _Ty> using EwiseMul = std::multiplies<_Ty>;
	template<typename _Ty> struct MatMul { 
		template<typename _ExprType> 
		MATRICE_GLOBAL_FINL _Ty operator() (const _Ty* lhs, const _ExprType& rhs, int _c) const; 
	};
	template<typename _Ty> struct Inverse { 
		MATRICE_GLOBAL_FINL _Ty* operator()(int M, _Ty* Out, _Ty* In = nullptr) const; 
	};
	};

	template<typename _ExprOp> struct Base_
	{
		using Derived = _ExprOp;
		template<typename _Ty> MATRICE_GLOBAL_INL operator Matrix<_Ty>()
		{
			Matrix<_Ty> ret(static_cast<Derived*>(this)->rows(), static_cast<Derived*>(this)->cols());
			return (ret = *static_cast<Derived*>(this));
		}
		template<typename _OutType> MATRICE_GLOBAL_INL _OutType eval() const
		{
			_OutType ret(static_cast<const Derived*>(this)->rows(), static_cast<const Derived*>(this)->cols());
			return (ret = *static_cast<const Derived*>(this));
		}
		template<typename _ROprd> MATRICE_GLOBAL_INL
		MatBinaryExpr<Derived, _ROprd, Op::MatMul<typename _ROprd::value_t>> mul(const _ROprd& _rhs)
		{
			return MatBinaryExpr<Derived, _ROprd, Op::MatMul<typename _ROprd::value_t>>(*static_cast<Derived*>(this), _rhs);
		}
		MATRICE_GLOBAL_INL double sum() const
		{
			double val = 0.0;
			int n = static_cast<const Derived*>(this)->size();
			for (int i = 0; i < n; ++i) 
				val += static_cast<const Derived*>(this)->operator()(i);
			return (val);
		}
		MATRICE_GLOBAL_INL std::size_t size() const { return M*N; }
		MATRICE_GLOBAL_INL std::size_t rows() const { return M; }
		MATRICE_GLOBAL_INL std::size_t cols() const { return N; }
	protected:
		std::size_t M, K, N;
	};
	template<class _T1, class _T2, typename _BinaryOp>
	class EwiseBinaryExpr : public Base_<EwiseBinaryExpr<_T1, _T2, _BinaryOp>>
	{
	public:
		using LOprd_t = _T1;
		using ROprd_t = _T2;
		using value_t = std::enable_if_t<std::is_same_v<typename _T1::value_t, typename _T2::value_t>, typename _T1::value_t>;

		MATRICE_GLOBAL_INL EwiseBinaryExpr(const LOprd_t& _lhs, const ROprd_t& _rhs) noexcept 
			: _LHS(_lhs), _RHS(_rhs) { 
			M = _LHS.rows(), N = _RHS.cols(); 
		}
		MATRICE_GLOBAL_INL EwiseBinaryExpr(const value_t& _scalar, const ROprd_t& _rhs) noexcept 
			: _Scalar(_scalar), _LHS(_rhs), _RHS(_rhs) { 
			M = _LHS.rows(), N = _RHS.cols(); 
		}

		MATRICE_GLOBAL_INL value_t operator() (const std::size_t _idx) const
		{
			if (std::numeric_limits<value_t>::infinity() == _Scalar)
				return _Optor(_LHS(_idx), _RHS(_idx));
			if (_Scalar != std::numeric_limits<value_t>::infinity())
				return _Optor(_Scalar, _RHS(_idx));
		}

	private:
		const value_t _Scalar = std::numeric_limits<value_t>::infinity();
		const LOprd_t& _LHS; const ROprd_t& _RHS;
		std::function<value_t(value_t, value_t)> _Optor = _BinaryOp();
		using Base_<EwiseBinaryExpr<_T1, _T2, _BinaryOp>>::M;
		using Base_<EwiseBinaryExpr<_T1, _T2, _BinaryOp>>::N;
	};
	template<class _T1, class _T2, typename _BinaryOp>
	class MatBinaryExpr : public Base_<MatBinaryExpr<_T1, _T2, _BinaryOp>>
	{
	public:
		using LOprd_t = _T1; using ROprd_t = _T2;
		using value_t = typename std::enable_if_t<std::is_same_v<typename _T1::value_t, typename _T2::value_t>, typename _T1::value_t>;

		MATRICE_GLOBAL_INL MatBinaryExpr(const LOprd_t& _lhs, const ROprd_t& _rhs) noexcept 
			: _LHS(_lhs), _RHS(_rhs) { 
			M = _LHS.rows(), K = _LHS.cols(), N = _RHS.cols(); 
		}

		MATRICE_HOST_INL value_t* operator() (value_t* res) const
		{
			return _Optor(_LHS.data(), _RHS.data(), res, M, K, N);
		}
		MATRICE_GLOBAL_INL value_t operator() (int r, int c) const
		{
			return _Optor(_LHS.ptr(r), _RHS, c);
		}
		MATRICE_GLOBAL_INL value_t operator() (const std::size_t _idx) const
		{
			int r = _idx / N, c = _idx - r * N;
			return _Optor(_LHS.ptr(r), _RHS, c);
		}
	private:
		const LOprd_t& _LHS; 
		const ROprd_t& _RHS;
		_BinaryOp _Optor;
		using Base_<MatBinaryExpr<_T1, _T2, _BinaryOp>>::M;
		using Base_<MatBinaryExpr<_T1, _T2, _BinaryOp>>::K;
		using Base_<MatBinaryExpr<_T1, _T2, _BinaryOp>>::N;
	};
	template<class _Oprd, typename _UnaryOp>
	class MatUnaryExpr : public Base_<MatUnaryExpr<_Oprd, _UnaryOp>>
	{
	public:
		using type = typename std::enable_if<std::is_class<_Oprd>::value, _Oprd>::type;
		using value_t = typename std::enable_if<std::is_arithmetic<typename _Oprd::value_t>::value, typename _Oprd::value_t>::type;

		MATRICE_GLOBAL_INL MatUnaryExpr(const _Oprd& mat) noexcept
			: _RHS(mat), _ANS(mat) { M = _RHS.rows(), N = _RHS.cols(); }
		MATRICE_GLOBAL_INL MatUnaryExpr(const _Oprd& _rhs, _Oprd& _ans) noexcept
			: _RHS(_rhs), _ANS(_ans) { M = _RHS.rows(), N = _RHS.cols(); }

		MATRICE_GLOBAL_INL value_t* operator()() const { 
			return _Optor(M, _ANS.data(), _RHS.data());
		}
		MATRICE_GLOBAL_INL const _Oprd& ans() const { 
			return _ANS; 
		}

		template<typename _ROprd> MATRICE_GLOBAL_INL
		MatBinaryExpr<_Oprd, _ROprd, Op::MatMul<value_t>> mul(const _ROprd& _rhs)
		{
			auto data = this->operator()();
			return MatBinaryExpr<_Oprd, _ROprd, Op::MatMul<value_t>>(_ANS, _rhs);
		}

	private:
		const _Oprd& _RHS;
		const _Oprd& _ANS;
		_UnaryOp _Optor;
		using Base_<MatUnaryExpr<_Oprd, _UnaryOp>>::M;
		using Base_<MatUnaryExpr<_Oprd, _UnaryOp>>::N;
	};
};
template<typename _Ty> template<typename _ExprType> MATRICE_GLOBAL_INL
_Ty Expr::Op::MatMul<_Ty>::operator() (const _Ty* lhs, const _ExprType& rhs, int _c) const
{
	_Ty val = _Ty(0);
	const int K = rhs.rows(), N = rhs.cols();
#ifdef __disable_simd__
	for (int k = 0; k < K; ++k) val += lhs[k] * rhs(k*N + _c);
#else
#ifdef __AVX__
	for (int k = 0; k < K; ++k) val += lhs[k] * rhs(k*N + _c);
#endif
#endif
	return (val);
}

template<typename _Ty> MATRICE_GLOBAL_INL
_Ty* Expr::Op::Inverse<_Ty>::operator() (int M, _Ty* Out, _Ty* In) const
{
	const auto A = In;
	auto Ainv = Out;

	_Ty _Det = A[0] * A[4] * A[8] + A[1] * A[5] * A[6] + A[2] * A[3] * A[7]
		- A[6] * A[4] * A[2] - A[7] * A[5] * A[0] - A[8] * A[3] * A[1];

	if ((_Det < 0 ? -_Det : _Det) < 1.E-6) return nullptr;

	_Det = 1. / _Det;
	volatile _Ty A0 = A[0], A1 = A[1], A2 = A[2];
	volatile _Ty A3 = A[3], A4 = A[4], A6 = A[6];

	Ainv[0] = (A[4] * A[8] - A[7] * A[5]) * _Det;
	Ainv[1] = -(A[1] * A[8] - A[7] * A[2]) * _Det;
	Ainv[2] = (A1   * A[5] - A[4] * A[2]) * _Det;

	Ainv[3] = -(A[3] * A[8] - A[6] * A[5]) * _Det;
	Ainv[4] = (A0   * A[8] - A[6] * A2) * _Det;
	Ainv[5] = -(A0   * A[5] - A3 * A2) * _Det;

	Ainv[6] = (A3 * A[7] - A[6] * A4) * _Det;
	Ainv[7] = -(A0 * A[7] - A6 * A1) * _Det;
	Ainv[8] = (A0 * A4 - A3 * A1) * _Det;

	return (Out);
}
MATRICE_NAMESPACE_EXPR_END