/*
 */
#pragma once

#include "erosion.cuh"
#include "python_helper.h"
#include <cstring>



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



    // ngd.def("set_water_map", [](Erosion &self, nb::ndarray<float> arr) {
    //     if (arr.ndim() != 2)
    //         throw std::runtime_error("Input must be a 2D float32 array");

    //     size_t total_size = arr.shape(0) * arr.shape(1);
    //     self.host_water_map.resize(total_size);
    //     std::memcpy(self.host_water_map.data(), arr.data(), total_size * sizeof(float));
    // });

    // safer with "c_contig" contiguous
    ngd.def("set_water_map", [](Erosion &self, nb::ndarray<float, nb::c_contig> arr) {
        if (arr.ndim() != 2)
            throw std::runtime_error("Input must be a 2D float32 array");

        size_t total_size = arr.shape(0) * arr.shape(1);
        self.host_water_map.resize(total_size);
        std::memcpy(self.host_water_map.data(), arr.data(), total_size * sizeof(float)); // needs #include <cstring> in linux
    });


    // ngd.def("get_water_map"






}

} // namespace erosion