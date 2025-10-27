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
#include "resample.bind.h"       // 🚧 untested
#include "shader_maps.bind.h"    // 🚧 adding a new AO
// #include "template_class.bind.h" // 🚧 template example
#include "template_class2.bind.h" // 🚧 new template



namespace nb = nanobind;

static void bind_hello(nb::module_ &m) {
    m.def("hello", []() { return "Hello from Python!"; });
}

static void bind_cuda_hello(nb::module_ &m) {
    m.def("cuda_hello", []() { cuda_hello(); });
}

NB_MODULE(cuda_texture_gen, m) {

    bind_hello(m);
    bind_cuda_hello(m);

    erosion::bind(m);
    blur::bind(m);
    noise_generator::bind(m);
    resample::bind(m);
    shader_maps::bind(m);

    // template_class::bind(m);
    template_class2::bind(m);
}
