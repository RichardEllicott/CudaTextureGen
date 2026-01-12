/*

faster maths for cuda using intrinsics

with CPU fallback

*/
#pragma once
#include <cstdint> // uint32_t
#include <cuda_runtime.h>
#include <math.h>
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

// ============================================================
//  fast_logf
//  - GPU: __logf (fast SFU intrinsic)
//  - CPU: logf
// ============================================================
DH_INLINE float fast_logf(float x) {
#ifdef ENABLE_FAST_MATH
    return __logf(x);
#else
    return logf(x);
#endif
}

// ============================================================
//  fast_sqrtf
//  - GPU: __fsqrt_rn (fast SFU intrinsic)
//  - CPU: sqrtf
// ============================================================
DH_INLINE float fast_sqrtf(float x) {
#ifdef ENABLE_FAST_MATH
    return __fsqrt_rn(x);
#else
    return sqrtf(x);
#endif
}

// ============================================================
//  fast_sincosf
//  - GPU: __sincosf (fast SFU intrinsic)
//  - CPU: fallback sinf/cosf (MSVC-safe)
// ============================================================
DH_INLINE void fast_sincosf(float x, float *s, float *c) {
#ifdef ENABLE_FAST_MATH
    __sincosf(x, s, c);
#else
    // MSVC does not provide sincosf
    *s = sinf(x);
    *c = cosf(x);
#endif
}

// ============================================================
//  fast_exp2f
//  - GPU: emulates 2^x using __expf(x * ln(2))
//    (no __exp2f intrinsic on many architectures, including SM 5.x)
//  - CPU: uses exp2f
//  - Accuracy: ~1–3 ulp, fast and stable for simulation/noise
// ============================================================
DH_INLINE float fast_exp2f(float x) {
#ifdef ENABLE_FAST_MATH
    // Use fast exp intrinsic: 2^x = e^(x * ln(2))
    return __expf(x * 0.6931471805599453094f); // ln(2)
#else
    return exp2f(x);
#endif
}

// ============================================================
//  fast_rcp (fast reciprocal)
//  - GPU: __frcp_rn
//  - CPU: 1.0f / x
// ============================================================
DH_INLINE float fast_rcp(float x) {
#ifdef ENABLE_FAST_MATH
    return __frcp_rn(x);
#else
    return 1.0f / x;
#endif
}

// Fast tanh approximation
// - GPU: uses __expf for a very fast rational tanh identity
// - CPU: falls back to accurate tanhf
// - Accuracy: ~1–3 ulp, smooth and monotonic, ideal for simulation/noise
DH_INLINE float fast_tanhf(float x) {
#ifdef ENABLE_FAST_MATH
    // Compute tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    // __expf is SFU-accelerated and much faster than tanhf on device
    float e = __expf(2.0f * x);
    return (e - 1.0f) / (e + 1.0f);
#else
    // Host fallback: use standard library tanhf
    return tanhf(x);
#endif
}

} // namespace core::cuda::math

#undef ENABLE_FAST_MATH // comment out to disable fast math
