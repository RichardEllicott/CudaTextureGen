/*

exposing the DeviceArray to python



*/
#pragma once

#include "macros.h"
#include "nanobind_helper.h"

namespace gna_device_array {

namespace nb = nanobind;

template <typename T, int Dim>
void bind_device_array(nb::module_ &m, const char *name) {
    using DeviceArray = core::cuda::DeviceArray<T, Dim>;

    // create class
    auto ngd = nb::class_<DeviceArray>(m, name).def(nb::init<>());

    // upload numpy array via python property
    auto get_array = [](DeviceArray &self) { return nb::helper::numpy::to_array(self); };
    auto set_array = [](DeviceArray &self, nb::ndarray<T, nb::c_contig> array) { nb::helper::numpy::to_device_array(array, self); };
    ngd.def_prop_rw("array", get_array, set_array, "DESCRIPTION");

    auto get_shape = [](DeviceArray &self) { return self.dimensions(); };
    ngd.def_prop_ro("shape", get_shape, "Array dimensions"); // shape()

    // size() => int
    ngd.def("size", &DeviceArray::size);

    // dev_ptr() => int
    auto get_dev_ptr = [](DeviceArray &self) { return (uintptr_t)self.dev_ptr(); };
    ngd.def("dev_ptr", get_dev_ptr); // dev_ptr() => int
}

inline void bind(nb::module_ &m) {

    // float 1D, 2D, 3D
    bind_device_array<float, 1>(m, "DeviceArrayFloat1D");
    bind_device_array<float, 2>(m, "DeviceArrayFloat2D");
    bind_device_array<float, 3>(m, "DeviceArrayFloat3D");

    // int 1D, 2D, 3D
    bind_device_array<int, 1>(m, "DeviceArrayInt1D");
    bind_device_array<int, 2>(m, "DeviceArrayInt2D");
    bind_device_array<int, 3>(m, "DeviceArrayInt3D");
}

} // namespace gna_device_array

#include "template_macro_undef.h"
