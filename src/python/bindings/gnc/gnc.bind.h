/*

binding for ALL the gnc modules

*/
#pragma once
#include "gnc/gnc_erosion.cuh"
#include "gnc/gnc_example.cuh"
#include "gnc/gnc_example2.cuh"
#include "gnc/gnc_noise.cuh"
#include "nanobind_helper.h"

// ================================================================================================================================
// [Class List]
// --------------------------------------------------------------------------------------------------------------------------------

// (NAME, PYTHON_NAME)
#define CLASS_NAMES                         \
    X(example::GNC_Example, GNC_Example)    \
    X(example2::GNC_Example2, GNC_Example2) \
    X(noise::GNC_Noise, GNC_Noise)          \
    X(erosion::GNC_Erosion, GNC_Erosion)

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

//
// VERSION TARGETS A PROP RAW
//
// template <typename T>
// nb::class_<T> bind_class(nb::module_ &m, const char *name) {
//     auto cls = nb::class_<T>(m, name).def(nb::init<>());

//     // dynamic properties
//     if constexpr (has_properties<T>::value) {
//         std::apply([&](auto... p) {
//             (cls.def_rw(p.name, p.member), ...);
//         },
//                    T::properties());
//     }

//     // process function
//     cls.def("process", [](T &self) { self.process(); });

//     return cls;
// }
//
//
//
//
// VERSION TARGETS A PROP SETTER

template <typename Class, typename MemberPtr>
void bind_one_property(nb::class_<Class> &cls,
                       const char *name,
                       MemberPtr member) {
    using Field =
        std::remove_reference_t<decltype(std::declval<Class>().*member)>;

    // // WORKING
    // cls.def_prop_rw(
    //     name,
    //     // getter
    //     [member](Class &self) -> Field & {
    //         return self.*member;
    //     },
    //     // setter
    //     [member](Class &self, const Field &value) {
    //         self.set_par(self.*member, value);
    //     });

    // with par name
    cls.def_prop_rw(
        name,
        // getter
        [member](Class &self) -> Field & {
            return self.*member; // return raw
        },
        // setter
        [member, name](Class &self, const Field &value) {
            printf("[GNC] parameter '%s' updated\n", name);
            self.set_par(self.*member, value); // hook setter
        });
}

template <typename T>
nb::class_<T> bind_class(nb::module_ &m, const char *name) {
    auto cls = nb::class_<T>(m, name).def(nb::init<>());

    if constexpr (has_properties<T>::value) {
        std::apply(
            [&](auto... p) {
                (bind_one_property<T>(cls, p.name, p.member), ...);
            },
            T::properties());
    }

    cls.def("process", [](T &self) { self.process(); });

    return cls;
}

// ================================================================================================================================
// [Bindings]
// --------------------------------------------------------------------------------------------------------------------------------

inline void bind(nb::module_ &m) {
#ifdef CLASS_NAMES
#define X(NAME, PYTHON_NAME) \
    bind_class<NAME>(m, EXPAND_AND_STRINGIFY(PYTHON_NAME));
    CLASS_NAMES
#undef X
#endif
}

} // namespace gnc

#undef CLASS_NAMES
