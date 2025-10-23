/*

template for a bound object that includes a macro to set up Parameters



configuring it to be all automatic:
https://copilot.microsoft.com/chats/2XJNp3jKPgdUwKbQoPHP1



*/
#pragma once

#include <cstdio>

#define CUDA_CHECK(call)                                                                               \
    do {                                                                                               \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

#define TEMPLATE_CLASS_PARAMETERS \
    X(int, width, 256)            \
    X(int, height, 256)           \
    X(float, test_par1, 0.0)      \
    X(float, test_par2, 1.0)      \
    X(float, test_par3, 1.0)

// #define TEMPLATE_CLASS_MAPS \
// X(int, data, 256)

#define TEMPLATE_CLASS_MAPS \
    X(float *, height_map)  \
    X(float *, blend_mask)  \
    X(float *, gradient_map)

namespace template_class {

struct Parameters {
    // declare pars on structures
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE NAME;
    TEMPLATE_CLASS_PARAMETERS
#undef X

// declare maps on structures
#define X(TYPE, NAME) TYPE NAME;
    TEMPLATE_CLASS_MAPS
#undef X
};

class TemplateClass {

  private:
    Parameters pars;
    Parameters *dev_pars;

    float *dev_data = nullptr;

  public:
    TemplateClass() {
        // set default pars
#define X(TYPE, NAME, DEFAULT_VAL) \
    pars.NAME = DEFAULT_VAL;
        TEMPLATE_CLASS_PARAMETERS
#undef X

        // null the maps
#define X(TYPE, NAME) pars.NAME = nullptr;
        TEMPLATE_CLASS_MAPS
#undef X
    }

// --------------------------------------------------------------------------------
// Decalare Get/Sets
// --------------------------------------------------------------------------------
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE get_##NAME() const;       \
    void set_##NAME(TYPE value);
    TEMPLATE_CLASS_PARAMETERS
#undef X
    // --------------------------------------------------------------------------------

    //
    //
    void allocate_memory(float *host_data, const unsigned width, const unsigned height) {

        free_memory();

        // cudaMalloc(&dev_pars, sizeof(Parameters));
        // cudaMemcpy(dev_pars, &pars, sizeof(Parameters), cudaMemcpyHostToDevice);

        // THIS PATTERN ADDS CHECKS!!!
        CUDA_CHECK(cudaMalloc(&dev_pars, sizeof(Parameters)));
        CUDA_CHECK(cudaMemcpy(dev_pars, &pars, sizeof(Parameters), cudaMemcpyHostToDevice));

        // data??
        cudaMalloc(&dev_data, width * height * sizeof(float));
        cudaMemcpy(dev_data, host_data, width * height * sizeof(float), cudaMemcpyHostToDevice);
    }

    void free_memory() {

        if (dev_data)
            cudaFree(dev_data);

        if (dev_pars)
            cudaFree(dev_pars);
    }
};

} // namespace template_class