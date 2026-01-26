/*

includes print functions

init_console() will ensure unicode support on Windows (for other print functions to)

print functions here

*/

#pragma once

#include <array>
#include <cstddef>
#include <iostream>
#include <sstream> // range_to_string
#include <string>

// block to support unicode in windows
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN // reduces stuff windows.h drags in
#define NOMINMAX
#include <windows.h>
#endif

namespace core::logging {

#pragma region EXPERIMENTAL

// convert any range like array to a string
template <typename Range>
std::string range_to_string(const Range &r) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (auto &x : r) {
        if (!first) oss << ", ";
        first = false;
        oss << x;
    }
    oss << "}";
    return oss.str();
}

#pragma endregion

// required for windows to show unicode strings with emojees
inline void init_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
}

// Base case print (not needed if you skip empty println)
static void print() {
    // nothing to print
}

// Array print (needs to be first for Linux?)
template <typename T, std::size_t N>
inline void print(const std::array<T, N> &arr) {
    std::cout << "{ ";
    for (std::size_t i = 0; i < N; ++i) {
        std::cout << arr[i];
        if (i + 1 < N) std::cout << ", ";
    }
    std::cout << " }";
}

// Single argument
template <typename T>
inline void print(const T &t) {
    std::cout << t;
}

// Variadic case
template <typename T, typename... Args>
inline void print(const T &t, const Args &...args) {
    std::cout << t;
    print(args...);
}

// With newline
template <typename... Args>
inline void println(const Args &...args) {
    print(args...);
    std::cout << std::endl;
}

// // ✅ New: printf-style (broken on linux)
// inline void printf(const char* fmt, ...) {
//     va_list args;
//     va_start(args, fmt);
//     std::vprintf(fmt, args);   // vprintf handles the varargs safely
//     va_end(args);
// }



// ================================================================================================================================
// // printf passthrough
// template <typename... Args>
// DH_INLINE void printf(const char *fmt, Args... args) {
// #ifdef __CUDA_ARCH__
//     // Device-side: use CUDA printf directly
//     printf(fmt, args...);
// #else
//     // Host-side: use standard printf
//     std::printf(fmt, args...);
// #endif
// }



} // namespace core::logging
