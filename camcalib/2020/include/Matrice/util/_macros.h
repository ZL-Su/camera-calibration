#pragma once

#ifndef __CXX11__
#define __CXX11__
#endif // !__CXX11__ defined for C++11 standard support

#ifndef __CXX17__
#define __CXX17__
#endif // !__CXX17__ defined for C++17 standard support

#if (_MSC_VER > 1600 || defined _HAS_CXX17)
#define DGE_USE_SHARED_MALLOC
#endif //enable shared memory allocator

#ifndef __enable_cuda__
//#define __enable_cuda__
#endif // !__enable_cuda__ defined for CUDA support

#ifndef __enable_ocv__
#define __enable_ocv__
#endif // !__enable_ocv__ defined for OpenCV view support

#ifdef _OPENMP
#define __enable_omp__
#endif // _OPENMP


#ifndef __AVX__
#define __AVX__
#endif // !__AVX__
#ifndef __SSE__
#define __SSE__
#endif // !__SSE__

#ifdef __enable_ocv__
#ifndef __use_ocv_as_view__
#define __use_ocv_as_view__
#endif
#include "../../../addin/ocv/include/opencv2/core.hpp"
#define VIEW_BASE_OCV cv::Mat
#endif // __enable_ocv__


#define MATRICE_NAMESPACE_BEGIN_ namespace Dgelo {
#define _MATRICE_NAMESPACE_END                   }
#define MATRICE_NAMESPACE_BEGIN_TYPES MATRICE_NAMESPACE_BEGIN_ namespace types {
#define MATRICE_NAMESPACE_END_TYPES  } _MATRICE_NAMESPACE_END
#define MATRICE_NAMESPACE_EXPR_BEGIN MATRICE_NAMESPACE_BEGIN_ namespace exprs {
#define MATRICE_NAMESPACE_EXPR_END MATRICE_NAMESPACE_END_TYPES 
