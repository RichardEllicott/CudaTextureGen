/*

includes print functions ... these functions printed emojee unicode on linux, but don't work on windows

*/

#pragma once

#include <iostream>
#include <string>

// block to support unicode in windows
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN // reduces stuff windows.h drags in
#define NOMINMAX
#include <windows.h>
#endif

namespace core::logging {

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



} // namespace core::logging

/*

vardic template used

template<typename... Args>
void printAll(Args... args) {
    (std::cout << ... << args) << "\n";  // fold expression (C++17)
}

int main() {
    printAll(10, "hello", 3.14);  // works with mixed types
}


*/