/*

dynamic properties base template using CRTP and constexpr for automatic binding

*/
#pragma once
// #include "template_macro_undef.h" // guard from defines

// #include <array> // std::array
#include <optional>
#include <stdexcept>   // exceptions
#include <tuple>       // std::tuple, std::make_tuple
#include <type_traits> // optional but often useful for traits
#include <utility>     // std::forward, std::index_sequence, etc.  std::swap

#include "core/cuda/cast.cuh" // top level for this object
#include "core/cuda/device_struct.cuh"
#include "core/cuda/hash.cuh"
#include "core/cuda/math.cuh"
#include "core/cuda/stream.cuh"
#include "core/cuda/types.cuh" // top level for this object
#include "macros.h"

#include "core/reflection.h"

#define GNC_BASE_LAZY_EVALUATION

// ================================================================================================================================

namespace gnc {

using namespace core::cuda::types; // include type aliases at top level
using namespace core::cuda::cast;  // include type aliases at top level
using namespace core::reflection;  // include at top level

namespace cmath = core::cuda::math; // include the cuda math lib as cmath
namespace chash = core::cuda::hash; // include the cuda math lib as cmath

#pragma region VALIDATORS // check if something is a particular type by template

// ================================================================================================================================
// Template Validators
// --------------------------------------------------------------------------------------------------------------------------------

// [core::Ref<core::cuda::DeviceArray<T, N>>]
template <typename T>
struct is_device_array_ref : std::false_type {};
template <typename T, int N>
struct is_device_array_ref<core::Ref<core::cuda::DeviceArray<T, N>>> : std::true_type {};

// --------------------------------------------------------------------------------------------------------------------------------

// [core::Ref<U>]
template <typename T>
struct is_ref : std::false_type {};
template <typename U>
struct is_ref<core::Ref<U>> : std::true_type {};

// ================================================================================================================================

#pragma endregion

struct NoParams {}; // default if we don't use the params

template <typename Derived, typename Parameters = NoParams, typename ArrayPointers = NoParams>
class GNC_Base {

  private:
    // CRTP: obtain this object as the concrete Derived type.
    Derived &derived() { return static_cast<Derived &>(*this); }
    const Derived &derived() const { return static_cast<const Derived &>(*this); }

  protected:
    // static core::reflection::Reflection<Derived> reflection; // using a generalize reflection pattern

    // ================================================================================================================================
  protected:
    Parameters _pars;
    ArrayPointers _arrays;

    core::cuda::DeviceStruct<Parameters> _dev_pars;      // uploads parameters struct to device
    core::cuda::DeviceStruct<ArrayPointers> _dev_arrays; // uploads array_pointers struct to device
    bool _pars_synced = false;

  public:
    core::Ref<core::cuda::Stream> stream; // gets a stream

    // ================================================================================================================================
    // [Set Par]
    // also marking the sync as dirty
    // --------------------------------------------------------------------------------------------------------------------------------
    // template for setting a par, is bound to python, will mark the _pars_synced
    template <typename T>
    void set_par(T &field, const T &value) {
        if (field != value) {
            // printf("value changed!\n");
            _pars_synced = false;
            field = value;
        }
    }
    // --------------------------------------------------------------------------------------------------------------------------------
    // opposite for clarity
    template <typename T>
    const T &get_par(const T &field) const {
        return field;
    }
    // ================================================================================================================================
    // [Ensure Array Ref Ready]
    // ensure the array ref is not empty, and if creating a new one set it up also marking sync as dirty
    // (ensuring pointers are uploaded before kernel launch)
    // --------------------------------------------------------------------------------------------------------------------------------
    // EXAMPLE:
    // ensure_array_ref_ready(water_map, height_map->shape(), true);
    // --------------------------------------------------------------------------------------------------------------------------------

    // simple helper to make ensuring an array exists with desired shape (if it doesn't already) quicker
    template <
        typename MapT,
        typename ShapeT,
        typename = std::enable_if_t<gnc::is_device_array_ref<MapT>::value>>
    inline void ensure_array_ref_ready(
        MapT &map, const ShapeT &desired_shape, bool zero_device = false) {

        map.instantiate_if_null(); // ensure the Ref is not empty

        if (map->shape() != desired_shape) { // if shape missmatch we will resize
            _pars_synced = false;            // and mark sync dirty (to trigger par upload)
            map->resize(desired_shape);
            if (zero_device) map->zero_device();
        }
    }

    // ================================================================================================================================

    // ready device ensuring par structs are uploaded
    void ready_device() {

        if (_pars_synced) return;     // skip if already synced
        stream.instantiate_if_null(); // ensure we have a stream

        _dev_pars.set_stream(stream->get()); // ensure streams
        _dev_arrays.set_stream(stream->get());

        derived()._ready_device(); // run derived which is set up by macro (currently copies all the vars to the pars by macro)

        _dev_pars.upload(_pars);     // upload pars
        _dev_arrays.upload(_arrays); // upload array pointers

        stream->sync();      // wait on stream, to ensure copying completes
        _pars_synced = true; // mark the pars as having synced, should prevent uploading unless pars have changed
    }

    // ================================================================================================================================
    // [constexpr Reflection]
    // --------------------------------------------------------------------------------------------------------------------------------
    // return properties

    // move to bottom

#pragma region COMPILE_TIME_REFLECTION // getting all of type, to do stuff like make all arrays valid etc

    // ================================================================================================================================
    // [OPTIONAL REFLECTION] return vector pointers to members whose type is exactly T
    // --------------------------------------------------------------------------------------------------------------------------------

    // get pointer list to all of type T
    // note used for multiple types could add more compile time overhead
    template <typename T>
    std::vector<T *> ct_get_all_of_type() {

        Derived &self = static_cast<Derived &>(*this); //
        std::vector<T *> result;
        auto props = Derived::properties();

        std::apply(
            [&](auto &...prop) {
                // process each property individually
                ([&](auto &p) {
                    using MemberT = decltype(self.*(p.member));
                    using DecayedMemberT = std::remove_reference_t<MemberT>;

                    if constexpr (std::is_same_v<DecayedMemberT, T>) {
                        result.push_back(&(self.*(p.member)));
                    }
                }(prop),
                 ...);
            },
            props);

        return result;
    }

    // --------------------------------------------------------------------------------------------------------------------------------
    // attempting to seperat the logic so it can be generalized

#pragma endregion

#pragma region RUNTIME_REFLECTION

#ifdef GNC_BASE_LAZY_EVALUATION

    // Returns the lazily‑constructed runtime property table for this class.
    // The table is built exactly once per *type*, on first use, and then cached.
    static const std::vector<RuntimeProperty> &runtime_properties() {
        static const std::vector<RuntimeProperty> props =
            build_runtime_properties_from_tuple<Derived>(Derived::properties());
        return props;
    }

    // lazy map
    static const std::unordered_map<std::string, RuntimeProperty> &runtime_property_map() {
        static const std::unordered_map<std::string, RuntimeProperty> map = [] {
            std::unordered_map<std::string, RuntimeProperty> m;
            m.reserve(runtime_properties().size());
            for (const auto &rp : runtime_properties()) {
                m.emplace(rp.name, rp);
            }
            return m;
        }();
        return map;
    }

    // ----------------------------------------------------------------


    

    // get all of type from the runtime_properties
    template <typename T>
    auto get_all_of_type() {
        using CleanT = std::remove_cv_t<std::remove_reference_t<T>>;

        std::vector<T *> result;
        auto &self = derived();

        for (auto const &rp : runtime_properties()) {
            if (*rp.type == typeid(CleanT)) {
                auto *raw_ptr = reinterpret_cast<char *>(&self) + rp.offset;
                auto *ptr = reinterpret_cast<T *>(raw_ptr);
                result.push_back(ptr);
            }
        }

        return result;
    }

#endif

#pragma endregion

#pragma region TESTS

#define REFACTOR_TO_REFLECTION_OB 1 // refactor to using the seperate Reflection object

    void _instance_test_1() {
        printf("_instance_test_1()...\n");

        for (auto *arr : ct_get_all_of_type<RefDeviceArrayFloat2D>()) {
            printf(" arr->instantiate_if_null()...\n");
            arr->instantiate_if_null();
        }
    }

    void _instance_test_2() {
        printf("_instance_test_2()...\n");

#if REFACTOR_TO_REFLECTION_OB == 0

        for (auto *arr : get_all_of_type<RefDeviceArrayFloat2D>()) {
            printf(" arr->instantiate_if_null()...\n");
            arr->instantiate_if_null();
        }

#elif REFACTOR_TO_REFLECTION_OB == 1

        auto &self = derived();

        for (auto *arr :
             core::reflection::Reflection<Derived>::template get_all_of_type<RefDeviceArrayFloat2D>(self)) {

            printf(" arr->instantiate_if_null()...\n");
            arr->instantiate_if_null();
        }

#endif
    }

    int _return_int_test(int v) {
        return v;
    }

    void _init_tests() {
    }

    // ================================================================================================================================

    // using get_all_of_type, ensure all arrays are instantiated (they will still be empty though)
    void test_inst_all_darrays() {

// (NAME)
#define REF_DEVICE_ARRAY_TYPES \
    X(RefDeviceArrayFloat1D)   \
    X(RefDeviceArrayFloat2D)   \
    X(RefDeviceArrayFloat3D)   \
    X(RefDeviceArrayInt1D)     \
    X(RefDeviceArrayInt2D)     \
    X(RefDeviceArrayInt3D)
#ifdef REF_DEVICE_ARRAY_TYPES
#define X(NAME)                                 \
    for (auto *arr : get_all_of_type<NAME>()) { \
        arr->instantiate_if_null();             \
    }
        REF_DEVICE_ARRAY_TYPES
#undef X
#endif
#undef REF_DEVICE_ARRAY_TYPES
    }

#pragma endregion

    GNC_Base() {
        _init_tests();

        // instantiate_all_arrays();
        stream.instantiate_if_null();
    }

    // main execution function, ensure device ready and run
    void compute() {
        // ready_device();      // ensure device ready
        derived()._compute();
    }

#pragma region REFLECTION_REGISTRY // properties and methods reflected at the end the class

    // return properties
    // note we must used "Derived" for properties
    static constexpr auto properties() {
        return std::tuple_cat(Derived::_properties(), // CRTP requirement
                              std::tuple{
                                  // ================================================================
                                  // [Default Properties]
                                  // ----------------------------------------------------------------
                                  Property<Derived, &Derived::stream>{"stream", &Derived::stream}, // trailing comma optional
                                  // ================================================================
                              });
    }

    // return properties UNUSED second store for testing
    static constexpr auto properties2() {
        return std::tuple_cat(Derived::_properties2(), // CRTP requirement
                              std::tuple{
                                  // ================================================================
                                  // [Default Properties]
                                  // ----------------------------------------------------------------
                                  // ================================================================
                              });
    }

    // return methods
    // ⚠️ note we MUST reference GNC_Base here for methods (not Derived)
    // this is slightly confusing as properties use Derived
    static constexpr auto methods() {
        return std::tuple_cat(Derived::_methods(), // CRTP requirement
                              std::tuple{
                                  // ================================================================
                                  // [Default Methods]
                                  // ----------------------------------------------------------------
                                  Method<GNC_Base, &GNC_Base::test_inst_all_darrays>{"test_inst_all_darrays"},
                                  Method<GNC_Base, &GNC_Base::_instance_test_1>{"_instance_test_1"},
                                  Method<GNC_Base, &GNC_Base::_instance_test_2>{"_instance_test_2"},
                                  Method<GNC_Base, &GNC_Base::_return_int_test>{"_return_int_test"},
                                  // ================================================================
                              });
    }

#pragma endregion
};

} // namespace gnc
