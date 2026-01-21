//
// ⚠️ THIS FILE IS COPIED OR GENERATED FROM 'gnc_template.cuh'
//

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

//     // ----------------------------------------------------------------
// // don't even need this?.... oh it was getting the type
// template <class T>
// struct is_device_array_type;

// template <class U, int N>
// struct is_device_array_type<core::Ref<core::cuda::DeviceArray<U, N>>> {
//     using type = core::cuda::DeviceArray<U, N>;
//     using value_type = U;
//     static constexpr int dims = N;
// };
// // ----------------------------------------------------------------

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

    // reflection string to member functions ⚠️ seems to be broken for the arrays atm, can't reflect these yet!
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

    // --------------------------------------------------------------------------------------------------------------------------------
    // 🧪 copy one property helper


    // COMPILES BUT NOT WHAT IS NEEDED
    // template <class Src, class Dst, class SrcProp, class DstTuple, std::size_t J = 0>
    // static inline void copy_property_by_name_impl(const Src &src,
    //                                               Dst &dst,
    //                                               const SrcProp &sp,
    //                                               const DstTuple &dst_props) {
    //     if constexpr (J < std::tuple_size_v<DstTuple>) {
    //         auto &dp = std::get<J>(dst_props);

    //         if (std::strcmp(sp.name, dp.name) == 0) {
    //             // Only copy if the types match exactly
    //             if constexpr (std::is_same_v<
    //                               decltype(src.*(sp.member)),
    //                               decltype(dst.*(dp.member))>) {
    //                 dst.*(dp.member) = src.*(sp.member);
    //             }
    //         }

    //         copy_property_by_name_impl<Src, Dst, SrcProp, DstTuple, J + 1>(src, dst, sp, dst_props);
    //     }
    // }

    // template <class Src, class Dst, class SrcProp>
    // static inline void copy_property_by_name(const Src &src,
    //                                          Dst &dst,
    //                                          const SrcProp &sp) {
    //     constexpr auto dst_props = Dst::properties();
    //     copy_property_by_name_impl(src, dst, sp, dst_props);
    // }


template <std::size_t I = 0, class Dst, class Value, class DstTuple>
static inline void set_property_by_name_impl(
    Dst& dst,
    const char* name,
    const Value& value,
    const DstTuple& dst_props)
{
    if constexpr (I < std::tuple_size_v<DstTuple>) {
        auto& dp = std::get<I>(dst_props);

        if (std::strcmp(dp.name, name) == 0) {
            // Only assign if the types match exactly
            if constexpr (std::is_same_v<
                              decltype(dst.*(dp.member)),
                              Value>)
            {
                dst.*(dp.member) = value;
            }
        }

        set_property_by_name_impl<I + 1>(dst, name, value, dst_props);
    }
}

template <class Dst, class Value>
static inline void set_property_by_name(Dst& dst,
                                        const char* name,
                                        const Value& value)
{
    constexpr auto dst_props = Dst::properties();
    set_property_by_name_impl(dst, name, value, dst_props);
}




    // --------------------------------------------------------------------------------------------------------------------------------
    // iterating all properties

    // 🧪 trying to iterate props, perform an action
    template <std::size_t I = 0, class Tuple, class F>
    static inline void for_each_property(const Tuple &props, F &&func) {
        if constexpr (I < std::tuple_size_v<Tuple>) {
            func(std::get<I>(props));
            for_each_property<I + 1>(props, std::forward<F>(func));
        }
    }

    // 🧪 testing iterating
    void test_iterate_props_to_get_arrays() {
        constexpr auto props = Self::properties();

        printf("test_iterate_props_to_get_arrays()...\n");

        for_each_property(props, [&](auto const &p) {
            printf("prop: %s\n", p.name);
            using MemberT = decltype(std::declval<Self>().*(p.member));
            using RawMemberT = std::remove_cv_t<std::remove_reference_t<MemberT>>;

            // is_ref
            if constexpr (is_ref<RawMemberT>::value) {
                printf("is_ref == true!\n");
                auto &ref = derived().*(p.member);
                ref.instantiate_if_null();
            }
            // is_device_array_ref
            if constexpr (is_device_array_ref<RawMemberT>::value) {
                // printf("is_device_array_ref == true!\n");
                // auto &value = derived().*(p.member); // should point to our member
            }

            // get actual device array
            if constexpr (is_device_array_ref<RawMemberT>::value) {

                // Access the actual Ref<DeviceArray<T,N>> instance
                auto &ref = derived().*(p.member);

                // ----------------------------------------------------------------
                // ⚠️ breaks GCC
                // core::RefBase &ref_base = ref; // for autotype
                // ref_base.instantiate_if_null();
                // ⚠️ also breaks GCC
                // core::RefBase &ref_base = static_cast<core::RefBase &>(ref);
                // ref_base.instantiate_if_null();
                // ----------------------------------------------------------------

                // Safety check (optional)
                if (!ref) {
                    printf("  %s is empty\n", p.name);
                    return;
                }

                // Now get the device pointer
                auto *ptr = ref->dev_ptr();

                set_property_by_name(_arrays, p.name, ptr);


                printf("  %s -> dev_ptr() = %p\n", p.name, (void *)ptr);
            }
        });
    }

    // --------------------------------------------------------------------------------------------------------------------------------

    // CRTP requirement
    void _ready_device() {

        test_iterate_props_to_get_arrays();

        //         // 🧪 testing DISABLED (handled by constexpr)
        //         // copy all pars to struct
        // #ifdef TEMPLATE_CLASS_PARAMETERS_STRUCT
        // #define X(TYPE, NAME, DEFAULT_VAL, DESCRIPTION) \
//     _pars.NAME = NAME;
        //         TEMPLATE_CLASS_PARAMETERS_STRUCT
        // #undef X
        // #endif

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
