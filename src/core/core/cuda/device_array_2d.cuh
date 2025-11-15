/*

2D version of DeviceArray, manages a device side array

*/
#pragma once

#include "core/cuda/device_array.cuh"

namespace core::cuda {

// 2D version of DeviceArray that store size
template <typename T>
class DeviceArray2D : public DeviceArray<T> {
  private:
    size_t _width = 0; // x
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
    size_t _width = 0; // x
    size_t _height = 0; // y
    size_t _depth = 0; // z

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

} // namespace core::cuda