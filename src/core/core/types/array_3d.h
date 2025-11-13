/*


Array2D and Array3D custom types

-see python_helper.h for functions to convert





Array Design Notes:


// Example: expose your Array2D to Python as a NumPy array without copying:

#include <nanobind/ndarray.h>
namespace nb = nanobind;

m.def("make_array", [](Array2D<double>& arr) {
    return nb::ndarray<nb::numpy, double>(
        arr.data(),
        {arr.get_height(), arr.get_width()},
        nb::handle() // you control ownership
    );
});




*/
#pragma once
// #include <cmath>
#include <vector>

namespace core::types {

// Special 3D array for volumes, row-major layout
template <typename T>
class Array3D {
    std::vector<T> _vector;
    size_t _width, _height, _depth;

  public:
    Array3D(size_t width = 0, size_t height = 0, size_t depth = 0)
        : _width(width), _height(height), _depth(depth),
          _vector(width * height * depth) {}

    // Mutable element access
    T &operator()(size_t z, size_t y, size_t x) {
        return _vector[(z * _height + y) * _width + x];
    }

    // Const element access
    const T &operator()(size_t z, size_t y, size_t x) const {
        return _vector[(z * _height + y) * _width + x];
    }

    // Raw pointer access
    T *data() { return _vector.data(); }
    const T *data() const { return _vector.data(); }

    // Size and dimensions
    size_t size() const { return _vector.size(); }
    bool empty() const { return _vector.empty(); }
    size_t get_width() const { return _width; }
    size_t get_height() const { return _height; }
    size_t get_depth() const { return _depth; }

    // Resize
    void resize(size_t width, size_t height, size_t depth) {
        _width = width;
        _height = height;
        _depth = depth;
        _vector.resize(_width * _height * _depth);
    }

    // Fill
    void clear(const T &value = T{}) {
        std::fill(_vector.begin(), _vector.end(), value);
    }

    // Flat indexing
    T &operator[](size_t i) { return _vector[i]; }
    const T &operator[](size_t i) const { return _vector[i]; }

    // Swap
    void swap(Array3D &other) noexcept {
        std::swap(_vector, other._vector);
        std::swap(_width, other._width);
        std::swap(_height, other._height);
        std::swap(_depth, other._depth);
    }
};

} // namespace core