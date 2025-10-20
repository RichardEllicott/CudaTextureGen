/*

CRTP Bindable test .... NOT WORKING.... RESEARCH for easy nanobind template

*/
#pragma once


/*
#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Base template: CRTP
template <typename Derived>
class Bindable {
    // Generic entry point: forwards to Derived::bind_impl
    static void bind(nb::class_<Derived> &cls) {
        Derived::bind_impl(cls);
    }
};

// //
// // 🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙
// //
// #include "bindable.hpp"

class ShaderMaps : public Bindable<ShaderMaps> {
    void generate_normal_map(float *in, float *out, int w, int h, float amount, bool wrap);
    void generate_ao_map(float *in, float *out, int w, int h, float radius, bool wrap);

    // Each class describes its own bindings here
    static void bind_impl(nb::class_<ShaderMaps> &cls) {

        cls.def("generate_normal_map", &ShaderMaps::generate_normal_map,
                nb::arg("in"), nb::arg("out"),
                nb::arg("w"), nb::arg("h"),
                nb::arg("amount") = 1.0f,
                nb::arg("wrap") = true)
            .def("generate_ao_map", &ShaderMaps::generate_ao_map,
                 nb::arg("in"), nb::arg("out"),
                 nb::arg("w"), nb::arg("h"),
                 nb::arg("radius") = 1.0f,
                 nb::arg("wrap") = true);
    }
};

// // 🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙🪙


*/