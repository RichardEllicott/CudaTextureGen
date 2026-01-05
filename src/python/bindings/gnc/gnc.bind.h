/*

binding for ALL the gnc modules

*/
#pragma once
#include "gnc/gnc_example.cuh"
#include "gnc/gnc_example2.cuh"
#include "nanobind_helper.h"

// ================================================================================================================================
// [Class List]
// --------------------------------------------------------------------------------------------------------------------------------

// (NAME)
#define CLASS_NAMES \
    X(GNC_Example)\
    X(GNC_Example2)

// ================================================================================================================================

namespace gnc {

// ================================================================================================================================
// [Binder Template]
// --------------------------------------------------------------------------------------------------------------------------------

// should detect properties()
// C++ 17 pattern (20 has neater syntax)
template <typename T, typename = void>
struct has_properties : std::false_type {};
template <typename T>
struct has_properties<T, std::void_t<decltype(T::properties())>> : std::true_type {};
// --------------------------------------------------------------------------------------------------------------------------------
template <typename T>
nb::class_<T> bind_class(nb::module_ &m, const char *name) {
    auto cls = nb::class_<T>(m, name).def(nb::init<>());

    // dynamic properties
    if constexpr (has_properties<T>::value) {
        std::apply([&](auto... p) {
            (cls.def_rw(p.name, p.member), ...);
        },
                   T::properties());
    }

    // process function
    cls.def("process", [](T &self) { self.process(); });

    return cls;
}

// ================================================================================================================================
// [Bindings]
// --------------------------------------------------------------------------------------------------------------------------------

inline void bind(nb::module_ &m) {
#ifdef CLASS_NAMES
#define X(NAME) \
    bind_class<NAME>(m, EXPAND_AND_STRINGIFY(NAME));
    CLASS_NAMES
#undef X
#endif
}

} // namespace gnc


#undef CLASS_NAMES

