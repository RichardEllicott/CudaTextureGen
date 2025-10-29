/*

custom types trying to match Godot's patterns (might not be used due to preferance for flat float arrays)

also special Array2D

*/

#pragma once

#include <cmath>
#include <vector>

// Move semantics: If

namespace core {

struct Vector2i {
    int x = 0;
    int y = 0;

    Vector2i() = default;
    Vector2i(int x, int y) : x(x), y(y) {}

    Vector2i operator+(const Vector2i &other) const { return {x + other.x, y + other.y}; }
    Vector2i operator-(const Vector2i &other) const { return {x - other.x, y - other.y}; }
    bool operator==(const Vector2i &other) const { return x == other.x && y == other.y; }
};

struct Vector3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vector3() = default;
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vector3 operator+(const Vector3 &other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vector3 operator-(const Vector3 &other) const { return {x - other.x, y - other.y, z - other.z}; }

    Vector3 normalized() const {
        float len = std::sqrt(x * x + y * y + z * z);
        return len > 0 ? Vector3{x / len, y / len, z / len} : Vector3{0, 0, 0};
    }
};

struct Color {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;

    Color() = default;
    Color(float r, float g, float b, float a = 1.0f) : r(r), g(g), b(b), a(a) {}
};

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
        _vector.resize(_width * _height);
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