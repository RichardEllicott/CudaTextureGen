/*

some generic macros

*/
#pragma once
// ================================================================================================================================
// [Device Only and Device/Host Convention]
// --------------------------------------------------------------------------------------------------------------------------------
#define D_INLINE __device__ __forceinline__           // device only functions
#define DH_INLINE __device__ __host__ __forceinline__ // device and host functions

// ================================================================================================================================
// [Dirty Error Check Call]
// --------------------------------------------------------------------------------------------------------------------------------

// calling cuda functions checking for errors (will report the line)
// quick and dirty
#define CUDA_CHECK(call)                                                                               \
    do {                                                                                               \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

// ================================================================================================================================
// [Macro Expand to Quoted String]
// --------------------------------------------------------------------------------------------------------------------------------
#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

// ================================================================================================================================
// ⚠️ EXPANSION TRICK BROKEN (trying to expand macros early)
// --------------------------------------------------------------------------------------------------------------------------------
// #define EXPAND(...) __VA_ARGS__
// #define EVAL(x) EXPAND(x)
// --------------------------------------------------------------------------------------------------------------------------------

// this trick allows merging defines, then undefing them
// if not use we will end up expanding the arrays later instead
// EXAMPLE:
/*
// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS_1   \
    X(float, 2, height_map, "")   \
    X(float, 2, water_map, "")

#define TEMPLATE_CLASS_ARRAYS_2         \
    X(float, 2, _water_out_map, "")     \
    X(float, 2, _sediment_out_map, "")

#define TEMPLATE_CLASS_ARRAYS     \
    EVAL(TEMPLATE_CLASS_ARRAYS_1) \
    EVAL(TEMPLATE_CLASS_ARRAYS_2)

#undef TEMPLATE_CLASS_ARRAYS_1
#undef TEMPLATE_CLASS_ARRAYS_2
*/

// ================================================================================================================================
