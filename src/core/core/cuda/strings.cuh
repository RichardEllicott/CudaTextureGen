/*

to_string overloads, allow converting everything to a string for printing

may add formatting helpers

*/
#pragma once

// #ifdef __CUDA_ARCH__
// #error "Formatting subsystem included in device code. This header is host-only."
// #endif

// #define EXTRAS // fmt

#include <array>         // std::array
#include <string>        // std::string, std::to_string
#include <tuple>         // std::tuple
#include <unordered_map> // std::unordered_map
#include <vector>        // std::vector

#include <map>
#include <optional>
#include <set>
#include <unordered_set>

#include "core/defines.h"

namespace core::strings {

// // forward declare to_string
// template <typename T>
// H_INLINE std::string to_string(const T &);

#pragma region STD_TO_STRING_PASSTHROUGH

// passthrough for int
H_INLINE std::string to_string(int v) {
    return std::to_string(v);
}

// passthrough for float
H_INLINE std::string to_string(float v) {
    return std::to_string(v);
}

// passthrough for double
H_INLINE std::string to_string(double v) {
    return std::to_string(v);
}

// passthrough for std::string
H_INLINE std::string to_string(const std::string &v) {
    return v;
}

// passthrough for const char*
H_INLINE std::string to_string(const char *v) {
    return std::string(v);
}

#pragma endregion

#pragma region CUDA_TYPES

// int2 → string
H_INLINE std::string to_string(const int2 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + "}";
    // return fmt("{{{}, {}}}", v.x, v.y);
}

// float2 → string
H_INLINE std::string to_string(const float2 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + "}";
    // return fmt("{{{}, {}}}", v.x, v.y);
}

// char2 → string
H_INLINE std::string to_string(const char2 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + "}";
    // return fmt("{{{}, {}}}", v.x, v.y);
}

// --------------------------------------------------------------------------------------------------------------------------------

// int3 → string
H_INLINE std::string to_string(const int3 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + "}";
    // return fmt("{{{}, {}, {}}}", v.x, v.y, v.z);
}

// float3 → string
H_INLINE std::string to_string(const float3 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + "}";
    // return fmt("{{{}, {}, {}}}", v.x, v.y, v.z);
}

// char3 → string
H_INLINE std::string to_string(const char3 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + "}";
    // return fmt("{{{}, {}, {}}}", v.x, v.y, v.z);
}

// --------------------------------------------------------------------------------------------------------------------------------

// int4 → string
H_INLINE std::string to_string(const int4 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " + std::to_string(v.w) + "}";
    // return fmt("{{{}, {}, {}, {}}}", v.x, v.y, v.z, v.w);
}

// float4 → string
H_INLINE std::string to_string(const float4 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " + std::to_string(v.w) + "}";
    // return fmt("{{{}, {}, {}, {}}}", v.x, v.y, v.z, v.w);
}

// char4 → string
H_INLINE std::string to_string(const char4 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " + std::to_string(v.w) + "}";
    // return fmt("{{{}, {}, {}, {}}}", v.x, v.y, v.z, v.w);
}

#pragma endregion

#pragma region HELPERS

#ifdef EXTRAS

// simple, minimal formatter replacing {} in order
template <typename... Args>
H_INLINE std::string fmt(const std::string &pattern, Args &&...args) {

    std::string out;
    out.reserve(pattern.size() + sizeof...(Args) * 8);

    const char *p = pattern.c_str();
    size_t arg_index = 0;
    std::array<std::string, sizeof...(Args)> arg_strings{
        to_string(std::forward<Args>(args))...};

    while (*p) {
        if (p[0] == '{' && p[1] == '}') {
            out += arg_strings[arg_index++];
            p += 2;
        } else {
            out += *p++;
        }
    }

    return out;
}

#endif
// --------------------------------------------------------------------------------------------------------------------------------

#pragma endregion

#pragma region COLLECTIONS

// join template for creating a list
template <typename Container>
H_INLINE std::string join(const Container &items, const std::string &sep) {
    std::string out;
    bool first = true;

    for (const auto &item : items) {
        if (!first)
            out += sep;
        first = false;

        out += to_string(item); // uses your overloads
    }

    return out;
}

// ================================================================================================================================

// vector → string
template <typename T>
H_INLINE std::string to_string(const std::vector<T> &items) {
    return "{" + join(items, ", ") + "}";
}
// --------------------------------------------------------------------------------------------------------------------------------

// std::array → string
template <typename T, std::size_t N>
H_INLINE std::string to_string(const std::array<T, N> &items) {
    return "{" + join(items, ", ") + "}";
}

// --------------------------------------------------------------------------------------------------------------------------------

// C‑style array → string
template <typename T, std::size_t N>
H_INLINE std::string to_string(const T (&items)[N]) {
    return "{" + join(items, ", ") + "}";
}

// --------------------------------------------------------------------------------------------------------------------------------

// std::unordered_map → string
template <typename K, typename V>
H_INLINE std::string to_string(const std::unordered_map<K, V> &m) {
    std::string out = "{";

    bool first = true;
    for (const auto &kv : m) {
        if (!first)
            out += ", ";
        first = false;

        out += to_string(kv.first);
        out += ": ";
        out += to_string(kv.second);
    }

    out += "}";
    return out;
}

// --------------------------------------------------------------------------------------------------------------------------------

// std::pair → string
template <typename A, typename B>
H_INLINE std::string to_string(const std::pair<A, B> &p) {
    return "{" + to_string(p.first) + ", " + to_string(p.second) + "}";
}

// --------------------------------------------------------------------------------------------------------------------------------

// std::map → string
template <typename K, typename V>
H_INLINE std::string to_string(const std::map<K, V> &m) {
    std::string out = "{";
    bool first = true;

    for (const auto &kv : m) {
        if (!first)
            out += ", ";
        first = false;

        out += to_string(kv.first);
        out += ": ";
        out += to_string(kv.second);
    }

    out += "}";
    return out;
}

// --------------------------------------------------------------------------------------------------------------------------------

// std::set → string
template <typename T>
H_INLINE std::string to_string(const std::set<T> &s) {
    return "{" + join(s, ", ") + "}";
}

// --------------------------------------------------------------------------------------------------------------------------------

// std::unordered_set → string
template <typename T>
H_INLINE std::string to_string(const std::unordered_set<T> &s) {
    return "{" + join(s, ", ") + "}";
}

// --------------------------------------------------------------------------------------------------------------------------------

// std::optional → string
template <typename T>
H_INLINE std::string to_string(const std::optional<T> &o) {
    if (!o) return "nullopt";
    return "optional(" + to_string(*o) + ")";
}

// --------------------------------------------------------------------------------------------------------------------------------

#if __cplusplus >= 202002L // C++20 or later

// std::span → string (C++ 20)
template <typename T>
H_INLINE std::string to_string(std::span<T> s) {
    return "{" + join(s, ", ") + "}";
}

#endif

// --------------------------------------------------------------------------------------------------------------------------------

// tuple helper
template <std::size_t I = 0, typename... Ts>
H_INLINE void tuple_to_string_parts(const std::tuple<Ts...> &t, std::string &out) {
    if constexpr (I < sizeof...(Ts)) {
        if (I > 0)
            out += ", ";
        out += to_string(std::get<I>(t));
        tuple_to_string_parts<I + 1>(t, out);
    }
}

// tuple → string
template <typename... Ts>
H_INLINE std::string to_string(const std::tuple<Ts...> &t) {
    std::string out = "{";
    tuple_to_string_parts(t, out);
    out += "}";
    return out;
}

#pragma endregion

// ================================================================================================================================

#pragma region DYNAMIC_TO_STRING_SUPPORT

// Detects whether T has a member function: std::string T::to_string() const
template <typename T>
class has_member_to_string {
  private:
    template <typename U>
    static auto test(int) -> decltype(std::declval<const U>().to_string(), std::true_type{});

    template <typename>
    static std::false_type test(...);

  public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

// If T has .to_string(), use it
template <typename T>
typename std::enable_if<has_member_to_string<T>::value, std::string>::type
to_string(const T &v) {
    return v.to_string();
}

#pragma endregion

// ================================================================================================================================

} // namespace core::strings
