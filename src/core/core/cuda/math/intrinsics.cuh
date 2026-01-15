/*

faster maths for cuda using intrinsics

with CPU fallback

used by the math library

*/
#pragma once
// #include <cstdint> // uint32_t
#include <cuda_runtime.h>
// #include <math.h>
// ========================================================================================================================
// [Options]
// ------------------------------------------------------------------------------------------------------------------------
#define ENABLE_FAST_MATH // comment out to disable fast math
// #pragma warning(disable:4068) // optional supress warnings
// ========================================================================================================================

#define DH_INLINE __device__ __host__ __forceinline__ // device and host functions

#if !defined(__CUDA_ARCH__)
#undef ENABLE_FAST_MATH // disable fast math on host side
#endif

// ========================================================================================================================

namespace core::cuda::math {

// fast ln(x)
DH_INLINE float fast_logf(float x) {
#ifdef ENABLE_FAST_MATH
    return __logf(x); // fast ln(x)
#else
    return logf(x); // precise host fallback
#endif
}

// ------------------------------------------------------------------------------------------------------------------------

// fast e^x
DH_INLINE float fast_expf(float x) {
#ifdef ENABLE_FAST_MATH
    return __expf(x); // fast e^x
#else
    return expf(x); // precise host fallback
#endif
}

// ------------------------------------------------------------------------------------------------------------------------

// no point in square root, it's not on older gpu's

// // fast sqrt(x)
// DH_INLINE float fast_sqrtf(float x) {
// #ifdef ENABLE_FAST_MATH
//     return __fsqrt_rn(x); // fast accurate (RN (IEEE))
//     // return __sqrtf(x); // faster, approx

// #else
//     return sqrtf(x);
// #endif
// }

// DH_INLINE float fast_sqrtf(float x) {
// #ifdef ENABLE_FAST_MATH
//     return sqrtf(x);        // let the compiler pick the best available
// #else
//     return sqrtf(x);
// #endif
// }

// ------------------------------------------------------------------------------------------------------------------------

// fast 1/sqrt(x)
DH_INLINE float fast_rsqrtf(float x) {
#ifdef ENABLE_FAST_MATH
    // return __frsqrt_rn(x); // accurate 1/sqrt(x)
    return rsqrtf(x); // fastest approximate 1/sqrt(x)
#else
    return 1.0f / sqrtf(x); // precise fallback
#endif
}

// ------------------------------------------------------------------------------------------------------------------------

// fast sine and cos
DH_INLINE void fast_sincosf(float x, float *sin, float *cos) {
#ifdef ENABLE_FAST_MATH
    __sincosf(x, sin, cos); // fast SFU sin/cos pair
#else
    *sin = sinf(x);
    *cos = cosf(x);
#endif
}

// ------------------------------------------------------------------------------------------------------------------------

// fast 2^x
DH_INLINE float fast_exp2f(float x) {
#ifdef ENABLE_FAST_MATH

// If the architecture is new enough to support __exp2f, use it.
// Pascal (SM 6.x) and newer support this intrinsic.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    return __exp2f(x);
#else
    // Fallback for older architectures: 2^x = e^(x * ln(2))
    return __expf(x * 0.6931471805599453094f);
#endif

#else
    return exp2f(x);
#endif
}

// ------------------------------------------------------------------------------------------------------------------------

// fast 1/x (reciprocal)
DH_INLINE float fast_rcpf(float x) {
#ifdef ENABLE_FAST_MATH
    return __frcp_rn(x); // fast 1/x
#else
    return 1.0f / x;
#endif
}


// ------------------------------------------------------------------------------------------------------------------------

// fast int → float (round-to-nearest)
DH_INLINE float fast_int2float(int x) {
#ifdef ENABLE_FAST_MATH
    return __int2float_rn(x); // hardware intrinsic
#else
    return static_cast<float>(x);
#endif
}






// ------------------------------------------------------------------------------------------------------------------------

// fast tanh
DH_INLINE float fast_tanhf(float x) {
#ifdef ENABLE_FAST_MATH
    // Compute tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    // __expf is SFU-accelerated and much faster than tanhf on device
    float exp2x = __expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
#else
    // Host fallback: use standard library tanhf
    return tanhf(x);
#endif
}

} // namespace core::cuda::math

#undef ENABLE_FAST_MATH // comment out to disable fast math
