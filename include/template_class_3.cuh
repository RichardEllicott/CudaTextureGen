/*

🧜‍♀️ TEMPLATE VERSION 20251027-3

this one uses more clever types and classes, needing less horrible macros

*/
#pragma once

// ════════════════════════════════════════════════ //
#define TEMPLATE_CLASS_NAME TemplateClass3
#define TEMPLATE_NAMESPACE template_class_3

#define TEMPLATE_CLASS_PARAMETERS \
    X(size_t, width, 256)         \
    X(size_t, height, 256)        \
    X(size_t, _block, 16)         \
    X(float, test, 0.0)

#define TEMPLATE_CLASS_MAPS \
    X(float, image)
// ════════════════════════════════════════════════ //

#include "cuda_types.cuh"

namespace TEMPLATE_NAMESPACE {

struct Parameters {
    // declare pars on structures
#define X(TYPE, NAME, DEFAULT_VAL) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
};

class TEMPLATE_CLASS_NAME {

  private:
    Parameters pars;

  public:
    // make getter/setters for the pars
#define X(TYPE, NAME, DEFAULT_VAL)                \
    TYPE get_##NAME() const { return pars.NAME; } \
    void set_##NAME(TYPE value) { pars.NAME = value; }
    TEMPLATE_CLASS_PARAMETERS
#undef X

// make maps
#define X(TYPE, NAME) \
    core::CudaArray2D<TYPE> NAME;
    TEMPLATE_CLASS_MAPS
#undef X

    void process();
};

} // namespace TEMPLATE_NAMESPACE
