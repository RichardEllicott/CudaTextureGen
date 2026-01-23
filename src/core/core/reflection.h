/*

trying to refactor general reflection patterns

*/
#pragma once

#include <typeinfo> // type_info
#include <unordered_map>
#include <vector>

namespace core::reflection {

#pragma region COMPILE_TIME_REFLECTION

// compile‑time description of a data member
template <typename T, typename Member>
struct _Property {
    const char *name;
    Member member;
};

// compile‑time helper alias (reduces boilerplate)
template <typename T, auto Member>
using Property = _Property<T, decltype(Member)>;

// compile‑time description of a member function
template <typename T, auto MethodPtr>
struct Method {
    const char *name;
    static constexpr auto member = MethodPtr;
    // if your property metadata exposes __this, add it too:
    // using This = T;
    // static constexpr auto __this = static_cast<T*>(nullptr);
};

#pragma endregion

#pragma region RUNTIME_REFLECTION

// structure to store runtime class property
struct RuntimeProperty {
    const char *name;
    size_t offset;
    const std::type_info *type; // NEW
};

// get memory offset of property
template <typename Class, typename Member>
constexpr size_t offset_of(Member Class::*m) {
    return reinterpret_cast<size_t>(
        &(reinterpret_cast<Class *>(0)->*m));
}

// build the runtime properties from the constexpr tuple
template <typename Derived, typename Tuple>
std::vector<RuntimeProperty> build_runtime_properties_from_tuple(const Tuple &props) {
    std::vector<RuntimeProperty> result;

    std::apply(
        [&](auto const &...prop) {
            (result.push_back(
                 RuntimeProperty{
                     prop.name,
                     offset_of(prop.member),                                                            // deduces Class automatically
                     &typeid(std::remove_reference_t<decltype(std::declval<Derived>().*(prop.member))>) // NEW
                 }),
             ...);
        },
        props);

    return result;
}

#pragma endregion

#pragma region REFLECTION_OBJECT

template <typename Derived>
class Reflection {

    // CRTP: obtain this object as the concrete Derived type.
    Derived &derived() { return static_cast<Derived &>(*this); }
    const Derived &derived() const { return static_cast<const Derived &>(*this); }

  public:
    // Returns the lazily‑constructed runtime property table for this class.
    // The table is built exactly once per *type*, on first use, and then cached.
    static const std::vector<RuntimeProperty> &runtime_properties() {
        static const std::vector<RuntimeProperty> props =
            build_runtime_properties_from_tuple<Derived>(Derived::properties());
        return props;
    }

    // lazy generated unordered_map
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

    // // runtime get all of type
    // template <typename T>
    // auto get_all_of_type() {
    //     using CleanT = std::remove_cv_t<std::remove_reference_t<T>>;

    //     std::vector<T *> result;
    //     auto &self = derived();

    //     for (auto const &rp : runtime_properties()) {
    //         if (*rp.type == typeid(CleanT)) {
    //             auto *raw_ptr = reinterpret_cast<char *>(&self) + rp.offset;
    //             auto *ptr = reinterpret_cast<T *>(raw_ptr);
    //             result.push_back(ptr);
    //         }
    //     }

    //     return result;
    // }

    // change to static?
    template <typename T>
    static std::vector<T *> get_properties_of_type(Derived &self) {
        std::vector<T *> result;

        auto props = Derived::properties();

        std::apply(
            [&](auto &...prop) {
                ([&](auto &p) {
                    using MemberT = decltype(self.*(p.member));
                    using Decayed = std::remove_reference_t<MemberT>;

                    if constexpr (std::is_same_v<Decayed, T>) {
                        result.push_back(&(self.*(p.member)));
                    }
                }(prop),
                 ...);
            },
            props);

        return result;
    }

#pragma region COMPILE_TIME_PATTERN

    // get pointer list to all of type T
    // note used for multiple types could add more compile time overhead
    template <typename T>
    auto ct_get_properties_of_type() {

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

#pragma endregion
};

#pragma endregion

} // namespace core::reflection