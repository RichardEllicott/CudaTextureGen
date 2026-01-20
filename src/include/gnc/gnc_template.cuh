/*

🧙‍♂️ gnc_template 20260115

contains master bolerplate that is copied via python script to gnc_boilerplate

*/
#pragma once
#include "_gnc_undef.h"
#include "template_macro_undef.h"

// ================================================================================================================================
// [Single Source of Truth]
// --------------------------------------------------------------------------------------------------------------------------------
#define TEMPLATE_CLASS_NAME GNC_Template
#define TEMPLATE_NAMESPACE gnc::_template // ❗ template is reserved

// must be trivially_copyable
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS_STRUCT             \
    X(bool, _debug, false, "")                       \
    X(int2, _size, {}, "")                           \
    X(int, tile_size, false, "for chequer_test")     \
    X(FloatArray<8>, float8, {}, "float array test") \
    X(IntArray<8>, int8, {}, "int array test")

// properties will not be added to the struct, eg Ref<> types work
// (TYPE, NAME, DEFAULT_VAL, DESCRIPTION)
#define TEMPLATE_CLASS_PARAMETERS           \
    X(RefDeviceArrayFloat2D, input, {}, "") \
    X(RefDeviceArrayFloat2D, output, {}, "")

// different pattern for arrays, allows better introspection
// (TYPE, DIMENSIONS, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_ARRAYS \
    X(float, 2, input2, "")   \
    X(float, 2, output2, "")

// optional class methods extra
// (TYPE, NAME, DESCRIPTION)
#define TEMPLATE_CLASS_METHODS \
    X(void, test, "test method")

#pragma region BOILERPLATE
// ================================================================================================================================
// [Boilerplate (all below can be cocidered a copy, should match)]
// --------------------------------------------------------------------------------------------------------------------------------
#include "gnc_base.cuh"

#ifndef TEMPLATE_CLASS_NAME
#error "TEMPLATE_CLASS_NAME must be defined before including this file"
#endif
#ifndef TEMPLATE_NAMESPACE
#error "TEMPLATE_NAMESPACE must be defined before including this file"
#endif

namespace TEMPLATE_NAMESPACE {
// ================================================================================================================================
// Parameters struct for uploading to GPU (UNUSED)
// --------------------------------------------------------------------------------------------------------------------------------
struct Parameters {
    using Self = Parameters;

#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

    // reflection string to member functions
    static constexpr auto properties() {
        return std::tuple{
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT // bind pars
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif
        };
    }
};
static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters must remain trivially copyable for CUDA memcpy");
// ================================================================================================================================
// ArrayDevicePointers struct for uploading to GPU (UNUSED)
// --------------------------------------------------------------------------------------------------------------------------------
struct ArrayPointers {
    using Self = ArrayPointers;

#ifdef TEMPLATE_CLASS_ARRAYS
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    TYPE *NAME;
    TEMPLATE_CLASS_ARRAYS
#undef X
#endif

    // reflection string to member functions
    static constexpr auto properties() {
        return std::tuple{
#ifdef TEMPLATE_CLASS_ARRAYS // bind pars
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_ARRAYS
#undef X
#endif
        };
    }
};
static_assert(std::is_trivially_copyable<ArrayPointers>::value, "ArrayPointers must remain trivially copyable for CUDA memcpy");
// ================================================================================================================================
// Main Class
// --------------------------------------------------------------------------------------------------------------------------------
class TEMPLATE_CLASS_NAME : public GNC_Base<TEMPLATE_CLASS_NAME, Parameters, ArrayPointers> {
    using Self = TEMPLATE_CLASS_NAME;

    // ================================================================
    // [Create pars and arrays]
    // ----------------------------------------------------------------

    // create pars
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

    // create arrays
#ifdef TEMPLATE_CLASS_PARAMETERS
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    TYPE NAME = DEFAULT_VAL;
    TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

    // create arrays2 (second pattern)
#ifdef TEMPLATE_CLASS_ARRAYS
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    core::Ref<core::cuda::DeviceArray<TYPE, DIMENSIONS>> NAME;
    TEMPLATE_CLASS_ARRAYS
#undef X
#endif

  public:
    // ================================================================
    // [Properties Binding]
    // ----------------------------------------------------------------

    // CRTP requirement
    static constexpr auto _properties() {
        return std::tuple{

#if REFACTOR_GNC_STORAGE_IN_PARS == 0

        // references by name of class parameters for reflection
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT // bind pars
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
            TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

#elif REFACTOR_GNC_STORAGE_IN_PARS == 1 // this was an attempt to bind to the par stucts properties... instead now building reflection to the pars ob

        // NestedProperty<Self, &Self::parameters, &Parameters::tile_size>{"tile_size"}, // tested working

#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT // bind pars
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    NestedProperty<Self, &Self::parameters, &Parameters::NAME>{EXPAND_AND_STRINGIFY(NAME)},
            TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

#endif

#ifdef TEMPLATE_CLASS_PARAMETERS // bind arrays
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
                TEMPLATE_CLASS_PARAMETERS
#undef X
#endif

#ifdef TEMPLATE_CLASS_ARRAYS // bind arrays2 (second pattern)
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    Property<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME), &Self::NAME},
                    TEMPLATE_CLASS_ARRAYS
#undef X
#endif
        };
    }
    // --------------------------------------------------------------------------------------------------------------------------------
    // CRTP requirement
    static constexpr auto _properties2() {
        return std::tuple{

        };
    }

    // ================================================================
    // [Method Binding]
    // ----------------------------------------------------------------

    // CRTP requirement
    static constexpr auto _methods() {
        return std::tuple{
        // Method<Self, &Self::test_method2>{"test_method2"},

#ifdef TEMPLATE_CLASS_METHODS // bind methods
#define X(TYPE, NAME, DESCRIPTION) \
    Method<Self, &Self::NAME>{EXPAND_AND_STRINGIFY(NAME)},
            TEMPLATE_CLASS_METHODS
#undef X
#endif

        };
    }

    // ================================================================================================================================
    // TRYNG TO GET THIS WORKING HERE BEFORE BASE
    // --------------------------------------------------------------------------------------------------------------------------------

    // // 🧪 testing
    // void __copy_properties(const Self &src, Parameters &dst) {

    //     constexpr auto src_props = Self::properties();
    //     constexpr auto dst_props = Parameters::properties();

    // }

    // --------------------------------------------------------------------------------------------------------------------------------

#pragma region WORKING_TEST_ON_DERIVED

    // // copy one source property into matching dst property by name
    // template <std::size_t J = 0, class SrcProp, class SrcType, class DstType, class DstTuple>
    // static inline void copy_one_dst(const SrcProp &sp,
    //                                 const SrcType &src,
    //                                 DstType &dst,
    //                                 const DstTuple &dst_props) {
    //     if constexpr (J < std::tuple_size_v<DstTuple>) {
    //         auto &dp = std::get<J>(dst_props);

    //         if (std::strcmp(sp.name, dp.name) == 0) {
    //             if constexpr (std::is_same_v<
    //                               decltype(src.*(sp.member)),
    //                               decltype(dst.*(dp.member))>) {
    //                 dst.*(dp.member) = src.*(sp.member);
    //             }
    //         }

    //         copy_one_dst<J + 1>(sp, src, dst, dst_props);
    //     }
    // }

    // // iterate all source properties
    // template <std::size_t I = 0, class SrcType, class DstType, class SrcTuple, class DstTuple>
    // static inline void copy_all_src(const SrcType &src,
    //                                 DstType &dst,
    //                                 const SrcTuple &src_props,
    //                                 const DstTuple &dst_props) {
    //     if constexpr (I < std::tuple_size_v<SrcTuple>) {
    //         auto const &sp = std::get<I>(src_props);
    //         copy_one_dst<>(sp, src, dst, dst_props);
    //         copy_all_src<I + 1>(src, dst, src_props, dst_props);
    //     }
    // }

    // // // inside your CRTP base / GNC_Template
    // // void __copy_properties(const Self &src, Parameters &dst) {
    // //     constexpr auto src_props = Self::properties();
    // //     constexpr auto dst_props = Parameters::properties();
    // //     copy_all_src(src, dst, src_props, dst_props);
    // // }

    // template <class Src, class Dst>
    // static inline void __copy_properties(const Src &src, Dst &dst) {
    //     constexpr auto src_props = Src::properties();
    //     constexpr auto dst_props = Dst::properties();
    //     copy_all_src(src, dst, src_props, dst_props);
    // }

#pragma endregion

    // --------------------------------------------------------------------------------------------------------------------------------

    // CRTP requirement
    void _ready_device() {

        // --------------------------------------------------------------------------------------------------------------------------------
        copy_properties(*this, _pars); // 🧪 testing

        // --------------------------------------------------------------------------------------------------------------------------------

        // copy all pars to struct
#ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
#define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
    _pars.NAME = NAME;
        TEMPLATE_CLASS_PARAMETERS_STRUCT
#undef X
#endif

        // copy all array pointers
#ifdef TEMPLATE_CLASS_ARRAYS // bind arrays2 (second pattern)
#define X(TYPE, DIMENSIONS, NAME, DESCRIPTION) \
    _arrays.NAME = nullptr;                    \
    if (NAME.is_valid()) {                     \
        _arrays.NAME = NAME->dev_ptr();        \
    }
        TEMPLATE_CLASS_ARRAYS
#undef X
#endif
    }
// --------------------------------------------------------------------------------------------------------------------------------
// bind extra methods
#ifdef TEMPLATE_CLASS_METHODS // bind arrays2 (second pattern)
#define X(TYPE, NAME, DESCRIPTION) \
    TYPE NAME();
    TEMPLATE_CLASS_METHODS
#undef X
#endif
    // --------------------------------------------------------------------------------------------------------------------------------

    void _compute(); // CRTP
};
} // namespace TEMPLATE_NAMESPACE

#undef REFACTOR_STORAGE_IN_PARS

#pragma endregion
