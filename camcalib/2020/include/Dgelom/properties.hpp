/**************************************************************************
This file is part of dgecalib, an effcient camera calibration library.
Copyright(C) 2017-2020, Zhilong(Dgelom) Su, all rights reserved.
**************************************************************************/
#pragma once

#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <mmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <iomanip>

#if defined __BORLANDC__
#  include <fastmath.h>
#elif defined __cplusplus
#  include <cmath>
#else
#  include <math.h>
#endif
#ifdef HAVE_TEGRA_OPTIMIZATION
#  include "tegra_round.hpp"
#endif

#ifndef DGE_INLINE
#define DGE_INLINE inline
#endif

#define PROPERTY(_T, name) \
__declspec(property(put = _set_##name, get = _get_##name)) _T name; \
typedef _T _type_##name

#define READONLY_PROPERTY(_T, name) \
__declspec(property(get = _get_##name)) _T name;\
typedef _T _type_##name

#define WRITEONLY_PROPERTY(_T, name) \
__declspec(property(put = _set_##name)) _T name;\
typedef _T _type_##name

#define __property__(_T, name) \
__declspec(property(put = _set_##name, get = _get_##name)) _T name; \
	typedef _T _type_##name

/*value returned type getter*/
#define __get__(name) inline _type_##name _get_##name()
/*copy assigned type setter*/
#define __set__(name) inline void _set_##name(_type_##name name)

#define CONST_PROPERTY(TYPE, NAME, EXP) \
__declspec(property(get = _Get_##NAME)) TYPE NAME;\
inline TYPE _Get_##NAME() const noexcept {\
return EXP;\
}

namespace dgelom {
	///<summary> 
	//@brief Template function to detect the data type
	//@author Zhilong (Dgelom) Su - May/15/2016 - SUTD 
	//@para Example-- _Ty* ty = nullptr; [_Ty* _Ty] = type_name<decltype(ty)>()
	///</summary>
	template <typename _Ty> DGE_INLINE std::string type_name() noexcept
	{
		using ty_nonref = typename std::remove_reference_t<_Ty>;

		std::unique_ptr<char, void(*)(void*)> uPtr(
#ifndef _MSC_VER
			abi::__cxa_demangle(typeid(ty_nonref).name(),
				nullptr, nullptr, nullptr),
#else
			nullptr,
#endif
			free);

		std::string strTyName = (uPtr != nullptr) ?
			uPtr.get() : typeid(ty_nonref).name();

		if constexpr (std::is_const<ty_nonref>::value)
			return "const " + strTyName;
		else if constexpr (std::is_volatile<ty_nonref>::value)
			return "volatile " + strTyName;
		else if constexpr (std::is_lvalue_reference<_Ty>::value)
			return strTyName + "&";
		else if constexpr (std::is_rvalue_reference<_Ty>::value)
			return strTyName + "&&";
		else
			return strTyName;
	};
	///<summary>
	//@brief: Template function to cvt. number to std::string
	//@author: Zhilong (Dgelom) Su - Jan.10.2017 @SEU
	///</summary>
	template <typename _Ty> DGE_INLINE std::string strf(_Ty _val) noexcept
	{
		std::ostringstream strs; strs << _val; return strs.str();
	}

	template <typename _Ty> DGE_INLINE int round_(_Ty _val) noexcept
	{
#if ((defined _MSC_VER && defined _M_X64) || (defined __x86_64__ \
    && defined __SSE2__)) && !defined(__CUDACC__)
		if constexpr (sizeof(_Ty) == 4) return _val;
		if constexpr (sizeof(_Ty) == 32) {
			__m128 t = _mm_set_ss((float)_val);
			return _mm_cvtss_si32(t);
		}
		if constexpr (sizeof(_Ty) == 64) {
			__m128d t = _mm_set_sd((double)_val);
			return _mm_cvtsd_si32(t);
		}
#elif (defined _MSC_VER && defined _M_IX86)
		int t;
		__asm
		{
			fld _val;
			fistp t;
		}
		return t;
#elif ((defined _MSC_VER && defined _M_ARM) && defined HAVE_TEGRA_OPTIMIZATION)
		TEGRA_ROUND_DBL(_val);
#else
		/* it's ok if round does not comply with IEEE754 standard;
		the tests should allow +/-1 difference when the tested functions use round */
		return (int)(_val + (_val >= 0 ? 0.5 : -0.5));
#endif
	}

	template<typename _Ty> DGE_INLINE int ceil_(_Ty value) noexcept
	{
#if (defined _MSC_VER && defined _M_X64) && !defined(__CUDACC__)
		if constexpr (sizeof(_Ty) == 4) return(value);
		if constexpr (sizeof(_Ty) == 32) {
			__m128 t = _mm_set_ss((float)value);
			int i = _mm_cvtss_si32(t);
			return i + _mm_movemask_ps(_mm_cmplt_ss(_mm_cvtsi32_ss(t, i), t));
		}
		if constexpr (sizeof(_Ty) == 64) {
			__m128d t = _mm_set_sd(value);
			int i = _mm_cvtsd_si32(t);
			return i + _mm_movemask_pd(_mm_cmplt_sd(_mm_cvtsi32_sd(t, i), t));
		}
#else
		int i = round_(value);
		float diff = (float)(i - value);
		return i + (diff < 0);
#endif
	}

	template<typename _Ty> DGE_INLINE int floor_(_Ty value) noexcept
	{
#if (defined _MSC_VER && defined _M_X64) && !defined(__CUDACC__)
		if constexpr (sizeof(_Ty) == 4) return(value);
		if constexpr (sizeof(_Ty) == 32) {
			__m128 t = _mm_set_ss((float)value);
			int i = _mm_cvtss_si32(t);
			return i - _mm_movemask_ps(_mm_cmplt_ss(_mm_cvtsi32_ss(t, i), t));
		}
		if constexpr (sizeof(_Ty) == 64) {
			__m128d t = _mm_set_sd(value);
			int i = _mm_cvtsd_si32(t);
			return i - _mm_movemask_pd(_mm_cmplt_sd(_mm_cvtsi32_sd(t, i), t));
		}
#else
		int i = round_(value);
		float diff = (float)(value - i);
		return i - (diff < 0);
#endif
	}

	template<typename _Inty> static inline
		/*convert matrix indices to linear index*/
		_Inty lidx(const _Inty _r, const _Inty _c, const _Inty _w)
	{
		return (_c + _r * _w);
	}
	template<typename _Inty> static inline
		/*convert linear index to matrix indices*/
		auto midx(const _Inty _lidx, const _Inty _w)
	{
		_Inty r = _lidx / _w;
		return std::make_tuple(_lidx - r * _w, r);
	}

	/*
	 * @Name: _HRC -- High Resolution Clock
	 * @Calling pipeline: _HRC.start -> "target block" -> _HRC_stop -> elapsed_time()
	 * @Copyright(c): Zhilong Su (su-zl@seu.edu.cn) 2017
	 */
	template<typename _Clock = std::chrono::high_resolution_clock> class _HRC final
	{
		typedef std::chrono::time_point<std::chrono::steady_clock> time_point;
	public:
		_HRC() { }
		~_HRC() { }

	public:
		// start-timer of high resolution clock
		__declspec(property(get = _prop_time_getter)) time_point start;
		// stop-timer of high resolution clock
		__declspec(property(get = _prop_time_getter)) time_point stop;
		inline time_point _prop_time_getter() { return _Clock::now(); }

		// return elapsed time
		inline double elapsed_time(const time_point& _start)
		{
			using namespace std;
			auto now = _Clock::now();
			auto interval = chrono::duration_cast<chrono::nanoseconds>(
				now - _start).count();
			m_time = (double)interval / m_ns2msconst;
			return m_time;
		}

		// output elasped time of runing target represented by _txt
		inline void elapsed_time(const time_point& _start, const std::string& _txt)
		{
			using namespace std;
			auto now = _Clock::now();
			auto interval = chrono::duration_cast<chrono::nanoseconds>(
				now - _start).count();
			m_time = (double)interval / m_ns2msconst;
			cout << " >> [Timer Message] Elapsed time of "
				<< _txt << setprecision(9)
				<< " " << m_time << "ms" << endl;
		}

	private:
		double m_time;
		const double m_ns2msconst = 1.0e6;
	};
}