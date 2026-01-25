/*

standard defines used for cuda

*/
#pragma once

// ================================================================================================================================

// Host-only compilation: define CUDA keywords away
#ifndef __CUDACC__

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __forceinline__
#define __forceinline__ inline
#endif

#endif

// ================================================================================================================================

// Device-only inline function
#define D_INLINE __device__ __forceinline__

// Host + device inline function
#define DH_INLINE __host__ __device__ __forceinline__

// Host-only inline function
#define H_INLINE __host__ __forceinline__

// ================================================================================================================================

// DH_CONST: device constant memory (CUDA) or constexpr (host)
// NOTE: Must be used at global scope when compiling with NVCC.
#ifdef __CUDACC__
#define DH_CONST __device__ __constant__
#else
#define DH_CONST constexpr
#endif
