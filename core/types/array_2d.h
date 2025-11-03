/*

template for an Array2D that uses a vector internally

see python_helper.h for functions to convert

*/
#pragma once
// #include <cmath>
#include <vector>

namespace core {

// special 2D array for maps, stores data in a std::vector in "row major" (row * width + col)
template <typename T>
class Array2D {
    std::vector<T> _vector;
    size_t _width, _height;

  public:
    Array2D(size_t width = 0, size_t height = 0)
        : _width(width), _height(height), _vector(width * height) {}

    // Mutable element access
    T &operator()(size_t row, size_t col) {
        return _vector[row * _width + col];
    }

    // Const element access
    const T &operator()(size_t row, size_t col) const {
        return _vector[row * _width + col];
    }

    // Raw pointer access
    T *data() { return _vector.data(); }
    const T *data() const { return _vector.data(); }

    // Size of the underlying flat array
    size_t size() const { return _vector.size(); }

    // Is empty (contains no data)
    bool empty() const { return _vector.empty(); }

    // Dimensions
    size_t get_width() const { return _width; }
    size_t get_height() const { return _height; }

    // Resize and reallocate
    void resize(size_t width, size_t height) {
        _width = width;
        _height = height;
        _vector.resize(_width * _height); // the memory is not going to change if we resize the same
    }

    // Fill the array with a value
    void clear(const T &value = T{}) {
        std::fill(_vector.begin(), _vector.end(), value);
    }

    // Flat indexing
    T &operator[](size_t i) { return _vector[i]; }
    const T &operator[](size_t i) const { return _vector[i]; }
};

} // namespace core