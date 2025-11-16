/*

2D version of DeviceArray, manages a device side array

*/
#pragma once

#include "device_array.cuh"

// working on DeviceArrayND ( a generic version)
#include <functional>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace core::cuda {

// 2D version of DeviceArray that store size
template <typename T>
class DeviceArray2D : public DeviceArray<T> {
  private:
    size_t _width = 0;  // x
    size_t _height = 0; // y

  public:
    DeviceArray2D() = default;

    DeviceArray2D(size_t width, size_t height) {
        resize(width, height);
    }

    void resize(size_t width, size_t height) {
        _width = width;
        _height = height;
        DeviceArray<T>::resize(width * height);
    }

    size_t width() const { return _width; }
    size_t height() const { return _height; }

    __host__ __device__
        size_t
        index(size_t x, size_t y) const {
        return y * _width + x; // row-major: y=row, x=col
    }

    // compiler will generate copy/move ctor/assign that call DeviceArray<T>'s versions
    // but you can explicitly default them for clarity:
    DeviceArray2D(const DeviceArray2D &) = default;
    DeviceArray2D &operator=(const DeviceArray2D &) = default;
    DeviceArray2D(DeviceArray2D &&) noexcept = default;
    DeviceArray2D &operator=(DeviceArray2D &&) noexcept = default;
    ~DeviceArray2D() = default;

    void swap(DeviceArray2D &other) noexcept {
        DeviceArray<T>::swap(other); // swap base
        std::swap(_width, other._width);
        std::swap(_height, other._height);
    }

    friend void swap(DeviceArray2D &a, DeviceArray2D &b) noexcept {
        a.swap(b);
    }

    // Delegate to parent upload
    void upload(const T *host_ptr, size_t width, size_t height) {
        resize(width, height);
        DeviceArray<T>::upload(host_ptr, width * height);
    }

    // Delegate to parent download
    void download(T *host_ptr) const {
        DeviceArray<T>::download(host_ptr);
    }
};

// 3D version of DeviceArray that stores dimensions
template <typename T>
class DeviceArray3D : public DeviceArray<T> {
  private:
    size_t _width = 0;  // x
    size_t _height = 0; // y
    size_t _depth = 0;  // z

  public:
    DeviceArray3D() = default;

    DeviceArray3D(size_t width, size_t height, size_t depth) {
        resize(width, height, depth);
    }

    void resize(size_t width, size_t height, size_t depth) {
        _width = width;
        _height = height;
        _depth = depth;
        DeviceArray<T>::resize(width * height * depth);
    }

    size_t width() const { return _width; }
    size_t height() const { return _height; }
    size_t depth() const { return _depth; }

    __host__ __device__
        size_t
        index(size_t x, size_t y, size_t z) const {
        // row-major order: z-slice, then row (y), then column (x)
        return z * (_width * _height) + y * _width + x;
    }

    // Explicitly default copy/move semantics
    DeviceArray3D(const DeviceArray3D &) = default;
    DeviceArray3D &operator=(const DeviceArray3D &) = default;
    DeviceArray3D(DeviceArray3D &&) noexcept = default;
    DeviceArray3D &operator=(DeviceArray3D &&) noexcept = default;
    ~DeviceArray3D() = default;

    void swap(DeviceArray3D &other) noexcept {
        DeviceArray<T>::swap(other); // swap base
        std::swap(_width, other._width);
        std::swap(_height, other._height);
        std::swap(_depth, other._depth);
    }

    friend void swap(DeviceArray3D &a, DeviceArray3D &b) noexcept {
        a.swap(b);
    }

    // Delegate to parent upload
    void upload(const T *host_ptr, size_t width, size_t height, size_t depth) {
        resize(width, height, depth);
        DeviceArray<T>::upload(host_ptr, width * height * depth);
    }

    // Delegate to parent download
    void download(T *host_ptr) const {
        DeviceArray<T>::download(host_ptr);
    }
};

// ⚠️ 🚧 no copy/move/swap yet.. working on it
template <typename T>
class DeviceArrayND : public DeviceArray<T> {

    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t total_size = 0;

  public:
    // Construct from initializer_list
    DeviceArrayND(std::initializer_list<size_t> dimensions) {
        throw std::runtime_error("DeviceArrayND not implemented yet");
        resize(dimensions);
    }

    // 1D
    DeviceArrayND(size_t length) {
        DeviceArrayND({length});
    }

    // 2D
    DeviceArrayND(size_t width, size_t height) {
        DeviceArrayND({width, height});
    }

    // 3D
    DeviceArrayND(size_t width, size_t height, size_t depth) {
        DeviceArrayND({width, height, depth});
    }

    void resize(std::initializer_list<size_t> dimensions) {
        shape.assign(dimensions);
        total_size = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        DeviceArray<T>::resize(total_size);

        // compute strides (row-major order)
        // strides are the 1D offset for the ND offset, like:
        // i = x + y * width
        // i = x + y * width + z * (width * height)
        strides.resize(shape.size());
        if (!shape.empty()) {
            strides.back() = 1;
            for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
    }

    // // Flatten ND indices into a 1D offset
    // size_t flatten(const std::vector<size_t>& indices) const {
    //     if (indices.size() != shape.size()) {
    //         throw std::runtime_error("Index dimensionality mismatch");
    //     }
    //     size_t idx = 0;
    //     for (size_t d = 0; d < indices.size(); ++d) {
    //         idx += indices[d] * strides[d];
    //     }
    //     return idx;
    // }

    // // Convenience operator() for up to 3D
    // T& operator()(size_t i) {
    //     return this->data()[flatten({i})];
    // }
    // T& operator()(size_t i, size_t j) {
    //     return this->data()[flatten({i,j})];
    // }
    // T& operator()(size_t i, size_t j, size_t k) {
    //     return this->data()[flatten({i,j,k})];
    // }
};

} // namespace core::cuda