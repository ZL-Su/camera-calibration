#pragma once

#include <xutility>
#include "..\util\_macros.h"

#ifdef __AVX__
#define MATRICE_ALIGN_BYTES   0x0020
#else
#ifdef __AVX512
#define MATRICE_ALIGN_BYTES   0x0040
#else
#define MATRICE_ALIGN_BYTES   0x0010
#endif
#endif // __AVX__

namespace Dgelo { namespace privt {

template<typename ValueType, typename IntegerType> ValueType* aligned_malloc(IntegerType size);
template<typename ValueType> void aligned_free(ValueType* aligned_ptr) noexcept;
template<typename ValueType> bool is_aligned(ValueType* aligned_ptr) noexcept;
template<typename ValueType, typename Integer> __forceinline 
ValueType* fill_mem(const ValueType* src, ValueType* dst, Integer size)
{
	if (size == 1) 
		dst[0] = src[0];
	if (size == 2)
		dst[0] = src[0], dst[1] = src[1];
	if (size == 3)
		dst[0] = src[0], dst[1] = src[1], dst[2] = src[2];
	if (size == 4)
		dst[0] = src[0], dst[1] = src[1], dst[2] = src[2], dst[3] = src[3];
	if (size == 5)
		dst[0] = src[0], dst[1] = src[1], dst[2] = src[2], dst[3] = src[3], dst[4] = src[4];
	if (size == 6)
		dst[0] = src[0], dst[1] = src[1], dst[2] = src[2], dst[3] = src[3], dst[4] = src[4], dst[5] = src[5];
	if (size == 7)
		dst[0] = src[0], dst[1] = src[1], dst[2] = src[2], dst[3] = src[3], dst[4] = src[4], dst[5] = src[5], dst[6] = src[6];
	if (size == 8)
		dst[0] = src[0], dst[1] = src[1], dst[2] = src[2], dst[3] = src[3], dst[4] = src[4], dst[5] = src[5], dst[6] = src[6], dst[7] = src[7];
	if (size == 9)
		dst[0] = src[0], dst[1] = src[1], dst[2] = src[2], dst[3] = src[3], dst[4] = src[4], dst[5] = src[5], dst[6] = src[6], dst[7] = src[7], dst[8] = src[8];
	if (size > 9) for (int i = 0; i < size; ++i) dst[i] = src[i];
	return (dst);
}
}
}


