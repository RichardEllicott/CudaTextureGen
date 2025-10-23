/*
 */
#pragma once

#include "erosion.cuh"
#include "python_helper.h"


namespace erosion{

namespace nb = nanobind;

inline static void bind(nb::module_ &m) {

    nb::class_<ErosionSimulator>(m, "ErosionSimulator")
        .def(nb::init<>())

        // .def_rw("rain_rate", &erosion::ErosionSimulator::rain_rate)
        // .def_rw("evaporation_rate", &erosion::ErosionSimulator::evaporation_rate)
        .def_rw("erosion_rate", &ErosionSimulator::erosion_rate)
        .def_rw("deposition_rate", &ErosionSimulator::deposition_rate)
        .def_rw("slope_threshold", &ErosionSimulator::slope_threshold)
        .def_rw("steps", &ErosionSimulator::steps)

// --------------------------------------------------------------------------------
// Declare CUDA constants
#define X(TYPE, NAME, DEFAULT_VAL) \
    .def_prop_rw(#NAME, &ErosionSimulator::get_##NAME, &ErosionSimulator::set_##NAME)
            EROSION_CONSTANTS
#undef X
        // --------------------------------------------------------------------------------

        // .def_prop_rw("max_height", &erosion::ErosionSimulator::get_max_height, &erosion::ErosionSimulator::set_max_height)
        // .def_prop_rw("min_height", &erosion::ErosionSimulator::get_min_height, &erosion::ErosionSimulator::set_min_height)

        .def("run_erosion", [](ErosionSimulator &self, nb::ndarray<float> arr) {
            if (arr.ndim() != 2)
                throw std::runtime_error("Input must be a 2D float32 array");

            int height = arr.shape(0);
            int width = arr.shape(1);
            float *data = arr.data();
            self.run_erosion(data, width, height);
        });
}


}