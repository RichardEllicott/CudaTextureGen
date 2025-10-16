/*

includes print functions

*/

#pragma once

#include <iostream>
#include <string>

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
