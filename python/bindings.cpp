/*

central file for the python bindings

concider rename to module.cpp

*/


// #include "python_helper.h" // python functions for dealing with numpy arrays etc
// #include <nanobind/nanobind.h>
// #include <nanobind/ndarray.h>
// #include <vector>


// tests and templates
#include "template_class_3.bind.h" // example template
#include "tests.bind.h"            // simple hello from gpu
#include "x_template_test.bind.h"  // auto generating template test

#include "blur.bind.h"            // 1D kernel based gaussian blur (passes horizontal and vertical)
#include "noise_generator.bind.h" // noise generators
#include "resample.bind.h"        // 🚧 untested
#include "shader_maps.bind.h"     // 🚧 adding a new AO

// erosion
#include "erosion2.bind.h" // simple erode with sediment only
#include "erosion3.bind.h" // introducing water for caving of river like patterns

namespace nb = nanobind;

NB_MODULE(cuda_texture_gen, m) {

    tests::bind(m);
    template_class_3::bind(m);
    x_template_test::bind(m);

    blur::bind(m);
    noise_generator::bind(m);
    resample::bind(m);
    shader_maps::bind(m);

    erosion_2::bind(m);
    erosion_3::bind(m);
}
