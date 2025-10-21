/*

central file for the python bindings

concider rename to module.cpp

*/

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

#include "python_helper.h"

#include "cuda_hello.cuh"

// new pattern
#include "blur.bind.h"
#include "erosion.bind.h"
#include "noise_generator.bind.h"
#include "resample.bind.h"    // 🚧 untested
#include "shader_maps.bind.h" // 🚧 adding a new AO

namespace nb = nanobind;

static void bind_hello(nb::module_ &m) {
    m.def("hello", []() { return "Hello from Python!"; });
}

static void bind_cuda_hello(nb::module_ &m) {
    m.def("cuda_hello", []() { cuda_hello(); });
}

NB_MODULE(cuda_hello, m) {

    bind_hello(m);
    bind_cuda_hello(m);

    bind_blur(m);
    bind_erosion(m);
    bind_noise_generator(m);
    bind_resample(m);
    bind_shader_maps(m);
}
