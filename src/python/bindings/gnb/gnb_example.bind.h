#pragma once
// #define BIND_DYNAMIC_PROPERTIES // prob not using this pattern but want to store it?

#include "gnb/gnb_example.cuh"
#include "nanobind_helper.h"
#include <any>

namespace TEMPLATE_NAMESPACE {

namespace nb = nanobind;

// ================================================================================================================================

#ifdef BIND_DYNAMIC_PROPERTIES

// binding pattern, supporting std::any
template <typename T>
void bind_dynamic_properties_any(
    nb::class_<T> &cls,
    bool (T::*has_property)(const std::string &) const,
    const std::any &(T::*get_property_any)(const std::string &) const,
    void (T::*set_property_any)(const std::string &, std::any)

) {
    cls.def("__getattr__", [=](nb::handle self, const char *key) -> nb::object {
        nb::object key_obj = nb::str(key);

        if (PyObject *result = PyObject_GenericGetAttr(self.ptr(), key_obj.ptr()))
            return nb::steal<nb::object>(result);

        PyErr_Clear();

        auto &cpp = nb::cast<T &>(self);
        if ((cpp.*has_property)(key)) {
            const std::any &v = (cpp.*get_property_any)(key);
            return nb::helper::any_to_nb_object(v);
        }

        PyErr_Format(PyExc_AttributeError, "No such attribute: %s", key);
        throw nb::python_error();
    });

    cls.def("__setattr__", [=](nb::handle self, const char *key, nb::object value) {
        nb::object key_obj = nb::str(key);

        if (PyObject_GenericSetAttr(self.ptr(), key_obj.ptr(), value.ptr()) == 0)
            return;

        PyErr_Clear();

        auto &cpp = nb::cast<T &>(self);
        (cpp.*set_property_any)(key, nb::helper::nb_object_to_any(value));
    });
}

#endif
// ================================================================================================================================

// should detect properties()
// C++ 17 pattern (20 has neater syntax)
template <typename T, typename = void>
struct has_properties : std::false_type {};
template <typename T>
struct has_properties<T, std::void_t<decltype(T::properties())>> : std::true_type {};

// attempt new template pattern
template <typename T>
nb::class_<T> bind_class(nb::module_ &m, const char *name) {
    auto cls = nb::class_<T>(m, name).def(nb::init<>());

    cls.def_rw("stream", &T::stream);
    cls.def("process", [](T &self) { self.process(); });

    // // C++ 20 can do this instead
    // if constexpr (requires { T::properties(); }) {
    //     std::apply([&](auto... p) {
    //         (cls.def_rw(p.name, p.member), ...);
    //     },
    //                T::properties());
    // }

    if constexpr (has_properties<T>::value) {
        std::apply([&](auto... p) {
            (cls.def_rw(p.name, p.member), ...);
        },
                   T::properties());
    }

    return cls;
}

// 🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪
// // could put this on the child objects
// static constexpr auto properties() {
//     return std::make_tuple(
//         Property<MyClass, decltype(&MyClass::heightmap)>{"heightmap", &MyClass::heightmap},
//         Property<MyClass, decltype(&MyClass::flow)>{"flow", &MyClass::flow}
//     );
// }
// 🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪🥪

inline void bind(nb::module_ &m) {

    // auto cls = nb::class_<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME)).def(nb::init<>());
    auto cls = bind_class<TEMPLATE_CLASS_NAME>(m, EXPAND_AND_STRINGIFY(TEMPLATE_CLASS_NAME));

    // ================================================================================================================================

//     // Ref<DeviceArray>'s
// #ifdef TEMPLATE_CLASS_DEVICE_ARRAYS
// #define X(TYPE, DIMENSIONS, DIM3, NAME, DESCRIPTION) \
//     cls.def_rw(EXPAND_AND_STRINGIFY(NAME), &TEMPLATE_CLASS_NAME::NAME);
//     TEMPLATE_CLASS_DEVICE_ARRAYS
// #undef X
// #endif

    // Ref<Stream>
    // cls.def_rw("stream", &TEMPLATE_CLASS_NAME::stream);

    // ================================================================================================================================
#ifdef BIND_DYNAMIC_PROPERTIES
    bind_dynamic_properties_any<GNB_Example>(
        cls,
        &GNB_Base::has_property,
        &GNB_Base::get_property_any,
        &GNB_Base::set_property_any);
#endif

    // ================================================================================================================================

    // cls.def("process", [](TEMPLATE_CLASS_NAME &self) {
    //     self.process();
    // });
}

} // namespace TEMPLATE_NAMESPACE