/*
 */
#pragma once

#include "erosion.cuh"
#include "python_helper.h"
#include <cstring> // required for std::memcpy in linux (not windows)

// standard macro for expanding to a string
#define STRINGIFY(x) #x
#define EXPAND_AND_STRINGIFY(x) STRINGIFY(x)

namespace erosion {

namespace nb = nanobind;

inline static void bind(nb::module_ &m) {

    auto ngd = nb::class_<Erosion>(m, "Erosion").def(nb::init<>());

    // .def_prop_rw("max_height", &erosion::ErosionSimulator::get_max_height, &erosion::ErosionSimulator::set_max_height)

#define X(TYPE, NAME, DEFAULT_VAL) \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), &Erosion::get_##NAME, &Erosion::set_##NAME);
    EROSION_PARAMETERS
#undef X

    ngd.def("run_erosion", [](Erosion &self, nb::ndarray<float> arr) {
        if (arr.ndim() != 2)
            throw std::runtime_error("Input must be a 2D float32 array");

        self.set_height(arr.shape(0));
        self.set_width(arr.shape(1));

        // int height = arr.shape(0);
        // int width = arr.shape(1);
        float *data = arr.data();
        self.run_erosion(data);
    });

    // 🚧 before template example
    // auto get_water_map = [](Erosion &self) { return python_helper::array2d_to_numpy_array(self.water_map); };
    // auto set_water_map = [](Erosion &self, nb::ndarray<float, nb::c_contig> arr) { self.water_map = python_helper::numpy_array_to_array2d(arr); };
    // ngd.def_prop_rw("water_map", get_water_map, set_water_map);

    // bind maps
#define X(TYPE, NAME)                                                                                                                      \
    auto get_##NAME = [](Erosion &self) { return python_helper::array2d_to_numpy_array(self.NAME); };                                      \
    auto set_##NAME = [](Erosion &self, nb::ndarray<float, nb::c_contig> arr) { self.NAME = python_helper::numpy_array_to_array2d(arr); }; \
    ngd.def_prop_rw(EXPAND_AND_STRINGIFY(NAME), get_##NAME, set_##NAME);
    EROSION_MAPS
#undef X
}

} // namespace erosion