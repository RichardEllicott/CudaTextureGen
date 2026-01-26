/*

binding for ALL the gnc modules

*/
#pragma once

#include "nanobind_helper.h"

#include "gnc/gnc_template.cuh" // master template, contains master boilerplate

// ================================================================================================================================
// [Modules]
// --------------------------------------------------------------------------------------------------------------------------------

#include "gnc/gnc_ao_map.cuh"
#include "gnc/gnc_drawing.cuh"
#include "gnc/gnc_example.cuh"
#include "gnc/gnc_noise.cuh"
#include "gnc/gnc_normal_map.cuh"
#include "gnc/gnc_resample.cuh"
#include "gnc/gnc_sea_erosion.cuh"
#include "gnc/gnc_slope_erosion.cuh"
#include "gnc/gnc_water.cuh"
#include "gnc/gnc_wind.cuh"

// #include "gnc/gnc_erosion.cuh"
#include "gnc/gnc_erosion2.cuh"

// (NAME, PYTHON_NAME)
#define CLASS_NAMES                                      \
    X(_template::GNC_Template, GNC_Template)             \
    X(example::GNC_Example, GNC_Example)                 \
    X(noise::GNC_Noise, GNC_Noise)                       \
    X(wind::GNC_Wind, GNC_Wind)                          \
    X(water::GNC_Water, GNC_Water)                       \
    X(resample::GNC_Resample, GNC_Resample)              \
    X(slope_erosion::GNC_SlopeErosion, GNC_SlopeErosion) \
    X(sea_erosion::GNC_SeaErosion, GNC_SeaErosion)       \
    X(normal_map::GNC_NormalMap, GNC_NormalMap)          \
    X(ao_map::GNC_AO_Map, GNC_AO_Map)                    \
    X(drawing::GNC_Drawing, GNC_Drawing)                 \
    X(erosion2::GNC_Erosion2, GNC_Erosion)

// X(erosion::GNC_Erosion, GNC_Erosion)

// ================================================================================================================================

#pragma region BOILERPLATE
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

template <typename Class, typename MemberPtr>
void bind_one_property(nb::class_<Class> &cls,
                       const char *name,
                       MemberPtr member) {
    using Field = std::remove_reference_t<decltype(std::declval<Class>().*member)>;

    cls.def_prop_rw(
        name,
        // getter
        [member](Class &self) -> Field & {
            return self.*member; // return raw
            // return self.get_par(self.*member); // can't seem to get this working???
        },
        // setter
        [member, name](Class &self, const Field &value) {
            // printf("[GNC] parameter '%s' updating...\n", name);
            self.set_par(self.*member, value); // hook setter
        });
}

// ================================================================================================================================

// 🧪 same as above but for method
template <typename, typename = void>
struct has_methods : std::false_type {};
template <typename T>
struct has_methods<T, std::void_t<decltype(T::methods())>> : std::true_type {};
// --------------------------------------------------------------------------------------------------------------------------------
// 🧪
template <typename T, auto MethodPtr>
void bind_one_method(nb::class_<T> &cls, const char *name) {
    cls.def(name, MethodPtr);
}
// --------------------------------------------------------------------------------------------------------------------------------

// generic bind a class called to bind a class that has:
// properties() - returns constexpr properties tuple
// methods() - returns constexpr methods tuple
// _compute() - launches compute
//
template <typename T>
nb::class_<T> bind_class(nb::module_ &m, const char *name) {
    auto cls = nb::class_<T>(m, name).def(nb::init<>());

    // ================================================================
    // [Bind Properties]
    // ----------------------------------------------------------------

    static_assert(has_methods<T>::value, "Class T must define static constexpr properties()"); // optional check
    if constexpr (has_properties<T>::value) {                                                  // optional check
        std::apply(
            [&](auto... p) {
                (bind_one_property<T>(cls, p.name, p.member), ...);
            },
            T::properties());
    }

    // ================================================================
    // [Bind Methods]
    // ----------------------------------------------------------------

    static_assert(has_methods<T>::value, "Class T must define static constexpr methods()"); // optional check
    if constexpr (has_methods<T>::value) {                                                  // optional check
        std::apply(
            [&](auto... mtd) {
                (bind_one_method<T, mtd.member>(cls, mtd.name), ...);
            },
            T::methods());
    }
    // ----------------------------------------------------------------

    cls.def("compute", [](T &self) { self._compute(); });

    return cls;
}

// ================================================================================================================================
// [Bindings]
// --------------------------------------------------------------------------------------------------------------------------------

inline void bind(nb::module_ &m) {
#ifdef CLASS_NAMES
#define X(NAME, PYTHON_NAME) \
    bind_class<NAME>(m, #PYTHON_NAME);
    CLASS_NAMES
#undef X
#endif
}

} // namespace gnc

#pragma endregion

#undef CLASS_NAMES
