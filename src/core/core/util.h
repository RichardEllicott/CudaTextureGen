/*



*/
#pragma once

#include <functional>    // std::invoke
#include <type_traits>   // std::decay_t, std::declval
#include <unordered_map> // std::unordered_map
// #include <utility>         // std::move (not strictly needed, but common)

namespace core::util {

#pragma region TO_UNORDERED_MAP

// Convert any iterable container (vector, array, span, etc.) into an unordered_map
// using a caller‑supplied key extractor (lambda, functor, or member accessor).
template <typename Vec, typename KeyFn>
auto to_unordered_map(const Vec &v, KeyFn key_fn) {
    using T = typename Vec::value_type;
    using Key = std::decay_t<decltype(std::invoke(key_fn, std::declval<const T &>()))>;

    std::unordered_map<Key, T> m;
    m.reserve(v.size());

    for (const auto &elem : v)
        m.emplace(std::invoke(key_fn, elem), elem);

    return m;
}

// Convenience overload: build an unordered_map keyed by elem.name
// for containers of structs that expose a .name field.
template <typename Vec>
auto to_unordered_map(const Vec &v) {
    using T = typename Vec::value_type;
    return to_unordered_map(v, [](const T &elem) { return elem.name; });
}

#pragma endregion

} // namespace core::util