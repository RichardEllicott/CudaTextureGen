/*
python bindings for Stream

⚠️ AI generated not yet checked
*/
#pragma once

#include "macros.h"
#include "nanobind_helper.h"

namespace stream {

namespace nb = nanobind;

inline void bind(nb::module_ &m) {
    using Stream = core::cuda::Stream;

    auto cls = nb::class_<Stream>(m, "Stream")
        // Constructor: optional flags
        .def(nb::init<unsigned int>(), nb::arg("flags") = cudaStreamDefault)

        // valid() -> bool
        .def("valid", &Stream::valid, "Check if the stream handle is valid")

        // get() -> int (uintptr_t)
        .def("handle", [](const Stream &self) {
            return (uintptr_t) self.get();
        }, "Return the underlying cudaStream_t as an integer")

        // sync()
        .def("sync", &Stream::sync, "Synchronize the CUDA stream")

        // __repr__
        .def("__repr__", [](const Stream &self) {
            if (self.valid())
                return "<Stream handle=" + std::to_string((uintptr_t)self.get()) + ">";
            else
                return "<Stream (null)>";
        });

    // No copy or deepcopy — Stream is move-only
    cls.def("__copy__", [](const Stream &) {
        throw std::runtime_error("Stream objects cannot be copied");
    });

    cls.def("__deepcopy__", [](const Stream &, nb::dict) {
        throw std::runtime_error("Stream objects cannot be deep-copied");
    });
}

} // namespace stream_binding

#include "template_macro_undef.h"
