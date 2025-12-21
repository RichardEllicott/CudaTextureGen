/*

central file for the python bindings

*/

// ================================================================
// tests and templates
// ----------------------------------------------------------------
#include "template_class_4.bind.h" //
#include "template_d_test.bind.h"  //  multi part template test 🚧🚧🚧🚧🚧 i broke it!
// #include "template_e_test.bind.h"   //  multi part template test 🚧🚧🚧🚧🚧 i broke it!
#include "template_darray_1.bind.h" //
#include "tests.bind.h"             // ✔️ simple hello from gpu
// #include "x_template_test.bind.h" //  auto generating template test // BROKEN!

// ================================================================
// image processing
// ----------------------------------------------------------------
#include "blur.bind.h"            // ✔️ 1D kernel based gaussian blur (passes horizontal and vertical)
#include "noise_generator.bind.h" // ✔️ noise generators, all support seamless noise!
#include "resample.bind.h"        // ✔️ resample mapping, can be used to say distort an image by noise
#include "shader_maps.bind.h"     // ✔️ 🚧 adding a new AO

#include "noise3d.bind.h" // ❌🚧 trying to make 3D rotatable noise.. didn't work, but refactors nice

// ================================================================
// erosion
// ----------------------------------------------------------------
// #include "erosion3.bind.h" // ✔️ introducing water for caving of river like patterns
// #include "erosion4.bind.h" // ✔️ simple erode refactored
// #include "erosion7.bind.h" // ✔️ advanced erosion with DeviceArray's
// #include "erosion8.bind.h" // 🚧 new features
#include "erosion10.bind.h" // 🐙 main version
#include "erosion9.bind.h"  // 🐙 main version

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

// ================================================================
// Nanobind compatability features
// ----------------------------------------------------------------
// allow std::array compatability to import/export
#include <array>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h> // ❗ allows std::array compatability

// ================================================================
// Nanobind compatability features
// ----------------------------------------------------------------

#include <nanobind/stl/shared_ptr.h>

#include "gna_device_array.bind.h"
#include "gna_graph_node.bind.h"

// ================================================================

// other options for nanobind support:
// #include <nanobind/stl/array.h>        // std::array
// #include <nanobind/stl/vector.h>       // std::vector
// #include <nanobind/stl/string.h>       // std::string
// #include <nanobind/stl/map.h>          // std::map, std::unordered_map
// #include <nanobind/stl/set.h>          // std::set, std::unordered_set
// #include <nanobind/stl/list.h>         // std::list
// #include <nanobind/stl/pair.h>         // std::pair
// #include <nanobind/stl/tuple.h>        // std::tuple
// #include <nanobind/stl/optional.h>     // std::optional
// #include <nanobind/stl/variant.h>      // std::variant
// #include <nanobind/stl/filesystem.h>   // std::filesystem::path
// #include <nanobind/stl/shared_ptr.h>   // std::shared_ptr
// #include <nanobind/stl/unique_ptr.h>   // std::unique_ptr
// #include <nanobind/stl/function.h>     // std::function
// #include <nanobind/stl/chrono.h>       // std::chrono types

// #include <nanobind/nanobind.h>
// #include <nanobind/stl/tuple.h>
// #include <cuda_runtime.h>

// namespace nanobind::detail {
//     template <>
//     struct type_caster<int2> {
//         NB_TYPE_CASTER(int2, const_name("tuple[int, int]"))

//         // Python -> C++ conversion
//         bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
//             // Try to convert from a tuple or list of 2 ints
//             if (PyTuple_Check(src.ptr()) && PyTuple_Size(src.ptr()) == 2) {
//                 PyObject *item0 = PyTuple_GetItem(src.ptr(), 0);
//                 PyObject *item1 = PyTuple_GetItem(src.ptr(), 1);

//                 value.x = PyLong_AsLong(item0);
//                 value.y = PyLong_AsLong(item1);

//                 if (PyErr_Occurred())
//                     return false;

//                 return true;
//             }
//             if (PyList_Check(src.ptr()) && PyList_Size(src.ptr()) == 2) {
//                 PyObject *item0 = PyList_GetItem(src.ptr(), 0);
//                 PyObject *item1 = PyList_GetItem(src.ptr(), 1);

//                 value.x = PyLong_AsLong(item0);
//                 value.y = PyLong_AsLong(item1);

//                 if (PyErr_Occurred())
//                     return false;

//                 return true;
//             }
//             return false;
//         }

//         // C++ -> Python conversion
//         static handle from_cpp(int2 src, rv_policy policy, cleanup_list *cleanup) noexcept {
//             return nb::make_tuple(src.x, src.y).release();
//         }
//     };
// }

// namespace nb = nanobind;

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
    gna_device_array::bind(m);
    gna_graph_node::bind(m);

}
