#pragma once

#include "nanobind_helper.h"

#include "gnb/gnb_example.cuh"

#include <any>

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

// ================================================================================================================================

// // binding pattern, supporting std::any
// template <typename T>
// void bind_dynamic_properties_any(
//     nb::class_<T> &cls,
//     bool (T::*has_property)(const std::string &) const,
//     const std::any &(T::*get_property_any)(const std::string &) const,
//     void (T::*set_property_any)(const std::string &, std::any)

// ) {
//     cls.def("__getattr__", [=](nb::handle self, const char *key) -> nb::object {
//         nb::object key_obj = nb::str(key);

//         if (PyObject *result = PyObject_GenericGetAttr(self.ptr(), key_obj.ptr()))
//             return nb::steal<nb::object>(result);

//         PyErr_Clear();

//         auto &cpp = nb::cast<T &>(self);
//         if ((cpp.*has_property)(key)) {
//             const std::any &v = (cpp.*get_property_any)(key);
//             return any_to_nb_object(v);
//         }

//         PyErr_Format(PyExc_AttributeError, "No such attribute: %s", key);
//         throw nb::python_error();
//     });

//     cls.def("__setattr__", [=](nb::handle self, const char *key, nb::object value) {
//         nb::object key_obj = nb::str(key);

//         if (PyObject_GenericSetAttr(self.ptr(), key_obj.ptr(), value.ptr()) == 0)
//             return;

//         PyErr_Clear();

//         auto &cpp = nb::cast<T &>(self);
//         (cpp.*set_property_any)(key, nb_object_to_any(value));
//     });
// }

// ================================================================================================================================

inline void bind(nb::module_ &m) {

    auto cls = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());

    // ================================================================================================================================

    // Ref<DeviceArray>'s
#ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
#define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
    cls.def_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::NAME);
    TEMPLATE_CLASS_DEVICE_ARRAYS
#undef X
#endif

    // Ref<Stream>
    cls.def_rw("stream", &TEMPLATE_CLASS_NAME::stream);

    // ================================================================================================================================

    // bind_dynamic_properties_any<GNB_Example>(
    //     cls,
    //     &GNB_Base::has_property,
    //     &GNB_Base::get_property_any,
    //     &GNB_Base::set_property_any);

    // ================================================================================================================================
    // [INLINE VERSION]

    cls.def("__getattr__", [](nb::handle self, const char *key) -> nb::object {
        nb::object key_obj = nb::str(key);

        // 1. Try Python/nanobind attributes first
        if (PyObject *result = PyObject_GenericGetAttr(self.ptr(), key_obj.ptr()))
            return nb::steal<nb::object>(result);

        PyErr_Clear();

        // 2. Try dynamic properties
        GNB_Example &cpp = nb::cast<GNB_Example &>(self);

        if (cpp.has_property(key)) {
            const std::any &v = cpp.get_property_any(key);
            return nb::helper::any_to_nb_object(v); // convert std::any → nb::object
        }

        // 3. Not found
        PyErr_Format(PyExc_AttributeError, "No such attribute: %s", key);
        throw nb::python_error();
    });

    cls.def("__setattr__", [](nb::handle self, const char *key, nb::object value) {
        nb::object key_obj = nb::str(key);

        // 1. Try Python/nanobind attributes first
        if (PyObject_GenericSetAttr(self.ptr(), key_obj.ptr(), value.ptr()) == 0)
            return;

        PyErr_Clear();

        // 2. Store dynamic property
        GNB_Example &cpp = nb::cast<GNB_Example &>(self);

        std::any converted = nb::helper::nb_object_to_any(value); // nb::object → std::any
        cpp.set_property_any(key, std::move(converted));
    });

    // ================================================================================================================================

    cls.def("process", [](TEMPLATE_CLASS_NAME &self) {
        self.process();
    });
}

} // namespace TEMPLATE_NAMESPACE