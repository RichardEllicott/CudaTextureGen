/*

central file for the python bindings

concider rename to module.cpp

*/

// #include "python_helper.h" // python functions for dealing with numpy arrays etc
// #include <nanobind/nanobind.h>
// #include <nanobind/ndarray.h>
// #include <vector>

// tests and templates
#include "template_class_3.bind.h" // ✔️ example template
#include "tests.bind.h"            // ✔️ simple hello from gpu
#include "x_template_test.bind.h"  // ✔️ auto generating template test

#include "blur.bind.h"            // ✔️ 1D kernel based gaussian blur (passes horizontal and vertical)
#include "noise_generator.bind.h" // ✔️ noise generators, all support seamless noise!
#include "resample.bind.h"        // ✔️ resample mapping, can be used to say distort an image by noise
#include "shader_maps.bind.h"     // ✔️ 🚧 adding a new AO

// erosion
#include "erosion2.bind.h" // ✔️ simple erode with sediment only
#include "erosion3.bind.h" // ✔️ introducing water for caving of river like patterns
#include "erosion4.bind.h" // ✔️ simple erode refactored
#include "erosion5.bind.h" // 🚧 erosion 3 working on changes

#include "fluid_simulation.bind.h" // 🚧

#include "tectonics.bind.h" // 🚧 testing the idea of propegating preassure waves to maybe create mountains or craters, also fluid sim here

#include "noise.bind.h" // ✔️ noise generators, all support seamless noise!

// #include "water_simulation.bind.h"

namespace nb = nanobind;

NB_MODULE(cuda_texture_gen, m) {

    tests::bind(m);
    template_class_3::bind(m);
    x_template_test::bind(m);

    blur::bind(m);
    noise_generator::bind(m);
    resample::bind(m);
    shader_maps::bind(m);

    erosion2::bind(m);
    erosion3::bind(m);
    erosion4::bind(m);
    erosion5::bind(m);

    fluid_simulation::bind(m);

    noise::bind(m);
}
