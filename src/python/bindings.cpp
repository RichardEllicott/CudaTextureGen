/*

central file for the python bindings

*/

// ================================================================
// [Tests]
// ----------------------------------------------------------------
#include "template_class_4.bind.h" //
#include "template_d_test.bind.h"  //  multi part template test 🚧🚧🚧🚧🚧 i broke it!
// #include "template_e_test.bind.h"   //  multi part template test 🚧🚧🚧🚧🚧 i broke it!
#include "template_darray_1.bind.h" //
#include "tests.bind.h"             // ✔️ simple hello from gpu
// #include "x_template_test.bind.h" //  auto generating template test // BROKEN!

// ================================================================
// [Image Processing]
// ----------------------------------------------------------------
#include "blur.bind.h"            // ✔️ 1D kernel based gaussian blur (passes horizontal and vertical)
#include "noise_generator.bind.h" // ✔️ noise generators, all support seamless noise!
#include "resample.bind.h"        // ✔️ resample mapping, can be used to say distort an image by noise
#include "shader_maps.bind.h"     // ✔️ 🚧 adding a new AO

#include "noise3d.bind.h" // ❌🚧 trying to make 3D rotatable noise.. didn't work, but refactors nice

// ================================================================
// [Erosion]
// ----------------------------------------------------------------
// #include "erosion3.bind.h" // ✔️ introducing water for caving of river like patterns
// #include "erosion4.bind.h" // ✔️ simple erode refactored
// #include "erosion7.bind.h" // ✔️ advanced erosion with DeviceArray's
// #include "erosion8.bind.h" // 🚧 new features
#include "erosion10.bind.h" // 🐙 main version
#include "erosion9.bind.h"  // 🐙 main version

// ================================================================
// [Experiments]
// ----------------------------------------------------------------
#include "fluid_simulation.bind.h" // 🚧
#include "tectonics.bind.h"        // 🚧 testing the idea of propegating preassure waves to maybe create mountains or craters, also fluid sim here

#include "misc.bind.h" //

// #ifdef _WIN32
// #include <windows.h> // windows only
// #endif

#include "core/logging.h"

// ================================================================
// Nanobind compatability features
// ----------------------------------------------------------------
// allow std::array compatability to import/export

// #include <nanobind/stl/array.h> // ❗ allows std::array compatability

// ================================================================
// [Graph Nodes]
// ----------------------------------------------------------------
#include "core/cuda/device_array.bind.h"
#include "gna_graph_node.bind.h"

#include "gna/gna_example.bind.h"
#include "gnb/gnb_example.bind.h"

#include "gnc/gnc.bind.h"

// ================================================================
// [Nanobind Options]
// ----------------------------------------------------------------
#include <nanobind/stl/array.h>  // std::array
#include <nanobind/stl/vector.h> // std::vector
// #include <nanobind/stl/string.h>       // std::string
// #include <nanobind/stl/map.h>          // std::map, std::unordered_map
// #include <nanobind/stl/set.h>          // std::set, std::unordered_set
// #include <nanobind/stl/list.h>         // std::list
// #include <nanobind/stl/pair.h>         // std::pair
// #include <nanobind/stl/tuple.h>        // std::tuple
// #include <nanobind/stl/optional.h>     // std::optional
#include <nanobind/stl/variant.h> // std::variant
// #include <nanobind/stl/filesystem.h>   // std::filesystem::path
#include <nanobind/stl/shared_ptr.h> // std::shared_ptr
// #include <nanobind/stl/unique_ptr.h>   // std::unique_ptr
// #include <nanobind/stl/function.h>     // std::function
// #include <nanobind/stl/chrono.h>       // std::chrono types
// ================================================================

NB_MODULE(cuda_texture_gen, m) {

    core::logging::init_console(); // ensure windows console supports unicode

    tests::bind(m);
    template_class_4::bind(m);
    template_d_test::bind(m);
    // template_e_test::bind(m);
    template_darray_1::bind(m);

    // x_template_test::bind(m); // BROKEN!

    blur::bind(m);
    noise_generator::bind(m);
    resample::bind(m);
    shader_maps::bind(m);

    noise3d::bind(m);

    // erosion4::bind(m);
    // erosion7::bind(m);
    // erosion8::bind(m);
    erosion9::bind(m);
    erosion10::bind(m);

    fluid_simulation::bind(m);
    tectonics::bind(m);
    misc::bind(m);

    // GNA
    device_array::bind(m);
    gna_graph_node::bind(m);
    gna_example::bind(m);
    gnb_example::bind(m);
    gnc::bind(m);



}
