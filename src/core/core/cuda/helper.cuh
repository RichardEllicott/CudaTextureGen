/*

*/
#pragma once

#include "core/defines.h"

namespace core::cuda::helper {

#ifdef __CUDACC__ // only use when compiling with NVCC

// get position in 1D kernel
D_INLINE int global_thread_pos1() {
    return int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
}

// get position in 2D kernel
D_INLINE int2 global_thread_pos2() {
    return make_int2(
        int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x),
        int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y));
}

// get position in 3D kernel
D_INLINE int3 global_thread_pos3() {
    return make_int3(
        int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x),
        int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y),
        int(blockIdx.z) * int(blockDim.z) + int(threadIdx.z));
}

#endif

} // namespace core::cuda::helper