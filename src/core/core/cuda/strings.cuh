/*


*/
#pragma once

#include <array>  // std::array
#include <string> // std::string, std::to_string
#include <tuple>  // std::tuple (if you're formatting tuples)
#include <vector> // std::vector

#include "core/defines.h"

#define QUALIFIERS H_INLINE // cuda host inline

namespace core::strings {

// ================================================================================================================================

// int2
QUALIFIERS std::string to_string(const int2 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + "}";
}

// int3
QUALIFIERS std::string to_string(const int3 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + "}";
}

// int4
QUALIFIERS std::string to_string(const int4 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " + std::to_string(v.w) + "}";
}

// float2
QUALIFIERS std::string to_string(const float2 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + "}";
}

// float3
QUALIFIERS std::string to_string(const float3 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + "}";
}

// float4
QUALIFIERS std::string to_string(const float4 &v) {
    return "{" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ", " + std::to_string(v.w) + "}";
}

// ================================================================================================================================

// passthrough for int
QUALIFIERS std::string to_string(int v) {
    return std::to_string(v);
}

// passthrough for float
QUALIFIERS std::string to_string(float v) {
    return std::to_string(v);
}

// passthrough for double
QUALIFIERS std::string to_string(double v) {
    return std::to_string(v);
}

// passthrough for std::string
QUALIFIERS std::string to_string(const std::string &v) {
    return v;
}

// passthrough for const char*
QUALIFIERS std::string to_string(const char *v) {
    return std::string(v);
}

// ================================================================================================================================

// join template for creating a list
template <typename Container>
QUALIFIERS std::string join(const Container &items, const std::string &sep) {
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

// vector to string
template <typename T>
QUALIFIERS std::string to_string(const std::vector<T> &items) {
    return "{" + join(items, ", ") + "}";
}
// --------------------------------------------------------------------------------------------------------------------------------

// std::array to string
template <typename T, std::size_t N>
QUALIFIERS std::string to_string(const std::array<T, N> &items) {
    return "{" + join(items, ", ") + "}";
}

// --------------------------------------------------------------------------------------------------------------------------------

// internal helper
template <std::size_t I = 0, typename... Ts>
QUALIFIERS void tuple_to_string_parts(const std::tuple<Ts...> &t, std::string &out) {
    if constexpr (I < sizeof...(Ts)) {
        if (I > 0)
            out += ", ";
        out += to_string(std::get<I>(t));
        tuple_to_string_parts<I + 1>(t, out);
    }
}

// tuple to string
template <typename... Ts>
QUALIFIERS std::string to_string(const std::tuple<Ts...> &t) {
    std::string out = "{";
    tuple_to_string_parts(t, out);
    out += "}";
    return out;
}

// ================================================================================================================================

} // namespace core::strings

#undef QUALIFIERS