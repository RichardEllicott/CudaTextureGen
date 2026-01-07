/*
python bindings for DeviceArray
*/
#pragma once

#include "macros.h"
#include "nanobind_helper.h"

namespace device_array {

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

    auto get_shape = [](DeviceArray &self) { return self.shape(); };
    ngd.def_prop_ro("shape", get_shape, "Array shape"); // shape()

    ngd.def("size", &DeviceArray::size); // size() => int

    // ngd.def("resize", (void (DeviceArray::*)(std::array<size_t, Dim>))&DeviceArray::resize); // brittle if we add overloads
    ngd.def("resize", [](DeviceArray &self, std::array<size_t, Dim> dims) { self.resize(dims); }); // lambda avoids overload issues

    // dev_ptr() => int
    auto get_dev_ptr = [](DeviceArray &self) { return (uintptr_t)self.dev_ptr(); };
    ngd.def("dev_ptr", get_dev_ptr); // dev_ptr() => int

    // copy
    ngd.def("__copy__", [](const DeviceArray &self) {
        return DeviceArray(self); // uses your copy constructor
    });

    ngd.def("__deepcopy__", [](const DeviceArray &self, nb::dict) {
        return DeviceArray(self); // deep copy
    });
}

inline void bind(nb::module_ &m) {

    // float 1-4D
    bind_device_array<float, 1>(m, "DeviceArrayFloat1D"); // core::cuda::DeviceArray<float, 1>
    bind_device_array<float, 2>(m, "DeviceArrayFloat2D"); // core::cuda::DeviceArray<float, 2>
    bind_device_array<float, 3>(m, "DeviceArrayFloat3D"); // core::cuda::DeviceArray<float, 3>
    bind_device_array<float, 4>(m, "DeviceArrayFloat4D"); // core::cuda::DeviceArray<float, 4>

    // int 1-4D
    bind_device_array<int, 1>(m, "DeviceArrayInt1D"); // core::cuda::DeviceArray<int, 1>
    bind_device_array<int, 2>(m, "DeviceArrayInt2D"); // core::cuda::DeviceArray<int, 2>
    bind_device_array<int, 3>(m, "DeviceArrayInt3D"); // core::cuda::DeviceArray<int, 3>
    bind_device_array<int, 4>(m, "DeviceArrayInt4D"); // core::cuda::DeviceArray<int, 4>
}

} // namespace device_array

#include "template_macro_undef.h"
