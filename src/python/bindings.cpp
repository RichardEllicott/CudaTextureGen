/*

central file for the python bindings

*/
// ================================================================
// tests and templates
// ----------------------------------------------------------------
#include "py_native_object_test.h" // ✔️ testing a more python based object
#include "template_class_4.bind.h" // ✔️ adding more template features
#include "template_darray_1.bind.h" // ✔️ adding more template features
#include "tests.bind.h"            // ✔️ simple hello from gpu
// #include "x_template_test.bind.h" //  auto generating template test // BROKEN!

// ================================================================
// image processing
// ----------------------------------------------------------------
#include "blur.bind.h"            // ✔️ 1D kernel based gaussian blur (passes horizontal and vertical)
#include "noise_generator.bind.h" // ✔️ noise generators, all support seamless noise!
#include "resample.bind.h"        // ✔️ resample mapping, can be used to say distort an image by noise
#include "shader_maps.bind.h"     // ✔️ 🚧 adding a new AO

#include "noise.bind.h" // 🚧 working on potential new noise generator but doesn't have all the same features yet

// ================================================================
// erosion
// ----------------------------------------------------------------
// #include "erosion3.bind.h" // ✔️ introducing water for caving of river like patterns
#include "erosion4.bind.h" // ✔️ simple erode refactored
#include "erosion5.bind.h" // ✔️ fairly developed new version (also can replicate simple erode)
#include "erosion6.bind.h" // 🚧 working on changes to previous

// ================================================================
// experiments
// ----------------------------------------------------------------
#include "fluid_simulation.bind.h" // 🚧
#include "tectonics.bind.h"        // 🚧 testing the idea of propegating preassure waves to maybe create mountains or craters, also fluid sim here

#include "misc.bind.h" //

// #ifdef _WIN32
// #include <windows.h> // windows only
// #endif

#include "core/logging.h"


namespace nb = nanobind;

NB_MODULE(cuda_texture_gen, m) {

    core::logging::init_console(); // ensure windows console supports unicode

    tests::bind(m);
    template_class_4::bind(m);
    template_darray_1::bind(m);
    py_native_object_test::PyNativeObjectTest::bind(m);

    // x_template_test::bind(m); // BROKEN!

    blur::bind(m);
    noise_generator::bind(m);
    resample::bind(m);
    shader_maps::bind(m);

    noise::bind(m);

    erosion4::bind(m);
    erosion5::bind(m);
    erosion6::bind(m);

    fluid_simulation::bind(m);
    tectonics::bind(m);
    misc::bind(m);
}
