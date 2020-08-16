#pragma once

#include "../include/Matrice/util/_macros.h"

#ifdef __use_ocv_as_view__
#ifdef __CXX11__
using ocv_view_t = VIEW_BASE_OCV;
template<typename Type>
using ocv_view_t_cast = cv::DataType<Type>;
#else
#define __view_space__ cv::
typedef cv::Mat ocv_view_t;
#endif
#endif // __use_ocv_as_view__

