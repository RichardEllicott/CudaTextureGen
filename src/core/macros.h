/*

not yet used? have some generic macros?


*/
#pragma once

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

// standard pattern to expan a define to a "string" (with the quote marks)
#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)