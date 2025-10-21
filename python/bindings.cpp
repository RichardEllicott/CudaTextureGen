/*

central file for the python bindings

concider rename to module.cpp

*/

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

#include "python_helper.h"

// #include "FastNoiseLite.h"

#include "blur.cuh"
#include "cuda_hello.cuh"
#include "erosion.cuh"
#include "noise_generator_d.cuh" // new noise techiques
#include "resample.cuh"          // 🚧 untested

// #include "shader_maps_c.cuh"     // 🚧 adding a new AO
#include "shader_maps.bind.h" // new pattern

namespace nb = nanobind;

static void bind_hello(nb::module_ &m) {
    m.def("hello", []() { return "Hello from Python!"; });
}

static void bind_cuda_hello(nb::module_ &m) {
    m.def("cuda_hello", []() { cuda_hello(); });
}

static void bind_noise_generator_d(nb::module_ &m) {

    auto ngd = nb::class_<noise_generator_d::NoiseGeneratorD>(m, "NoiseGeneratorD")
                   .def(nb::init<>())

    // bind get/sets
#define X(TYPE, NAME, DEFAULT_VAL) \
    .def_prop_rw(#NAME, &noise_generator_d::NoiseGeneratorD::get_##NAME, &noise_generator_d::NoiseGeneratorD::set_##NAME)
                       NOISE_GENERATOR_D_PARAMETERS
#undef X

                   .def("fill", [](noise_generator_d::NoiseGeneratorD &self, nb::ndarray<float> arr) {

                if (arr.ndim() != 2)
    throw std::runtime_error("Expected a 2D float32 array");


        int h = arr.shape(0);
        int w = arr.shape(1);
        float *data = arr.data();

        self.fill(data, w, h); });

    // Type enumerators
    nb::enum_<noise_generator_d::NoiseGeneratorD::Type>(ngd, "Type")

#define X(NAME) \
    .value(#NAME, noise_generator_d::NoiseGeneratorD::Type::NAME)
        NOISE_GENERATOR_D_TYPES
#undef X
            .export_values();
}

static void bind_erosion(nb::module_ &m) {

    nb::class_<erosion::ErosionSimulator>(m, "ErosionSimulator")
        .def(nb::init<>())

        // .def_rw("rain_rate", &erosion::ErosionSimulator::rain_rate)
        // .def_rw("evaporation_rate", &erosion::ErosionSimulator::evaporation_rate)
        .def_rw("erosion_rate", &erosion::ErosionSimulator::erosion_rate)
        .def_rw("deposition_rate", &erosion::ErosionSimulator::deposition_rate)
        .def_rw("slope_threshold", &erosion::ErosionSimulator::slope_threshold)
        .def_rw("steps", &erosion::ErosionSimulator::steps)

// --------------------------------------------------------------------------------
// Declare CUDA constants
#define X(TYPE, NAME, DEFAULT_VAL) \
    .def_prop_rw(#NAME, &erosion::ErosionSimulator::get_##NAME, &erosion::ErosionSimulator::set_##NAME)
            EROSION_CONSTANTS
#undef X
        // --------------------------------------------------------------------------------

        // .def_prop_rw("max_height", &erosion::ErosionSimulator::get_max_height, &erosion::ErosionSimulator::set_max_height)
        // .def_prop_rw("min_height", &erosion::ErosionSimulator::get_min_height, &erosion::ErosionSimulator::set_min_height)

        .def("run_erosion", [](erosion::ErosionSimulator &self, nb::ndarray<float> arr) {
            if (arr.ndim() != 2)
                throw std::runtime_error("Input must be a 2D float32 array");

            int height = arr.shape(0);
            int width = arr.shape(1);
            float *data = arr.data();
            self.run_erosion(data, width, height);
        });
}

static void bind_blur(nb::module_ &m) {

    m.def("blur", [](nb::ndarray<float, nb::ndim<2>, nb::c_contig> arr, float amount, bool wrap) {
        int h = arr.shape(0);
        int w = arr.shape(1);
        blur::blur(arr.data(), w, h, amount, wrap); }, nb::arg("arr"), nb::arg("amount"), nb::arg("wrap") = false);
}

static void bind_resample(nb::module_ &m) {
}

NB_MODULE(cuda_hello, m) {

    bind_hello(m);
    bind_cuda_hello(m);
    bind_noise_generator_d(m);
    bind_erosion(m);

    bind_shader_maps(m);
    bind_blur(m);

    bind_resample(m);
    bind_shader_maps(m);

    // python_helper::bind_python_helper(m);
}
