/*

DeviceArrayN

may be a more generic template


will have a common interface of "DeviceArrayNBase"



*/
#pragma once

// #include <cstddef>        // size_t
#include <cuda_runtime.h> // cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaError_t
// #include <stdexcept>      // std::runtime_error
#include <utility> // std::swap (needed for your swap implementation)
// #include <vector>         // std::vector<T>

// #include <iostream>
// #include <memory> // smart pointer

#include <array>

namespace core::cuda {

class DeviceArrayNBase {

    // cudaStream_t _stream = nullptr;

  protected:
    cudaStream_t _stream{nullptr}; // accessible to derived classes

  public:
    void set_stream(cudaStream_t stream) {
        _stream = stream;
    }

    // the total array size
    virtual size_t size() const = 0;
    // array size in bytes
    virtual size_t size_bytes() const = 0;
    // free the device, will also set the dimensions to 0 (which is the same as freeing the device)
    virtual void free_device() = 0;
    // initialize device memory to 0s (only if the size() > 0)
    virtual void zero_device() = 0;
    // is empty
    bool empty() const { return size() == 0; }

    //
    virtual ~DeviceArrayNBase() = default;
};

template <typename T, int Dim>
class DeviceArrayN : public core::cuda::DeviceArrayNBase {

    std::array<size_t, Dim> _dimensions{}; // default dimensions will be 0

    T *_dev_ptr = nullptr;

    // allocate device is private as this is achieved by resizing for the user
    void allocate_device() {
        free_device();

        if (size() == 0) // skip if size is 0
            return;

        auto err = cudaMallocAsync(&_dev_ptr, size_bytes(), _stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMallocAsync failed");
        }
    }

  public:
    DeviceArrayN() noexcept {
    }

    ~DeviceArrayN() override {
        free_device();
    }

    // the total array size
    size_t size() const override {
        size_t total = 1;
        for (auto d : _dimensions) total *= d;
        return total;
    }

    // array size in bytes
    size_t size_bytes() const override {
        return sizeof(T) * size();
    }

    // free the device, will also set the dimensions to 0 (which is the same as freeing the device)
    void free_device() override {
        if (_dev_ptr) {
            auto err = cudaFreeAsync(_dev_ptr, _stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaFreeAsync failed");
            }
            _dev_ptr = nullptr;
            _dimensions = {};
        }
    }

    T *dev_ptr() {
        return _dev_ptr;
    }

    void resize(std::array<size_t, Dim> dimensions) {
        if (_dimensions == dimensions) {
            return; // nothing changed, skip reallocation
        }

        free_device();
        _dimensions = dimensions;
        allocate_device();
    }

    void zero_device() override {

        if (!_dev_ptr) // no allocated memory (we just pass with no error for now)
            return;

        // concider error, throw std::runtime_error("zero_device called with size == 0");

        cudaError_t err = cudaMemsetAsync(_dev_ptr, 0, size_bytes(), _stream);

        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed");
        }
    }

    // Upload from host pointer
    void upload(const T *host_ptr, std::array<size_t, Dim> dimensions) {

        resize(dimensions); // resize, will reallocate if the size changes
        if (size() == 0)
            return;

        cudaError_t err = cudaMemcpyAsync(_dev_ptr, host_ptr, size_bytes(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMemcpy (Host->Device) failed");
    }

    // Download to host pointer (dangerous if the sizes don't match!)
    void download(T *host_ptr) const {
        if (!_dev_ptr || size() == 0)
            return;
        auto err = cudaMemcpyAsync(host_ptr, _dev_ptr, size_bytes(), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (Device->Host) failed");
        }
    }

#pragma region SWAP
    // swap member function
    void swap(DeviceArrayN<T, Dim> &other) noexcept {
        using std::swap;
        swap(_dev_ptr, other._dev_ptr);
        swap(_stream, other._stream);
        swap(_dimensions, other._dimensions);
    }

    // freind is a syntatic sugar to avoid putting the non-member function outside
    // this allows std::swap(myArrayA, myArrayB) to work
    friend void swap(DeviceArrayN<T, Dim> &a, DeviceArrayN<T, Dim> &b) noexcept {
        a.swap(b);
    }
#pragma endregion
#pragma region COPY

    // COPY
    DeviceArrayN(const DeviceArrayN &other) {
        _dimensions = other._dimensions;
        _stream = other._stream;
        _dev_ptr = nullptr;

        if (other._dev_ptr) {
            auto err = cudaMallocAsync(&_dev_ptr, size_bytes(), _stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMallocAsync failed in copy ctor");
            }

            // resize(_dimensions); // 🚧 we might reduce logic around here

            err = cudaMemcpyAsync(_dev_ptr, other._dev_ptr,
                                  size_bytes(), cudaMemcpyDeviceToDevice, _stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemcpyAsync failed in copy ctor");
            }
        }
    }

    DeviceArrayN &operator=(const DeviceArrayN &other) {
        if (this != &other) {
            DeviceArrayN tmp(other); // copy‑construct
            swap(tmp);               // strong exception safety
        }
        return *this;
    }

#pragma endregion

#pragma region MOVE

    DeviceArrayN(DeviceArrayN &&other) noexcept {
        swap(other);
        other._dev_ptr = nullptr;
        other._dimensions = {};
        other._stream = nullptr;
    }

    DeviceArrayN &operator=(DeviceArrayN &&other) noexcept {
        if (this != &other) {
            free_device(); // release current
            swap(other);
            other._dev_ptr = nullptr;
            other._dimensions = {};
            other._stream = nullptr;
        }
        return *this;
    }

#pragma endregion
};

template <typename T>
class DeviceArrayN2D : public DeviceArrayN<T, 2> {
  public:
    using Base = DeviceArrayN<T, 2>;
    using Base::Base; // inherit constructors

    DeviceArrayN2D() = default;

    // convenience accessors
    size_t width() const { return this->_dimensions[0]; }
    size_t height() const { return this->_dimensions[1]; }

    void resize(size_t w, size_t h) {
        Base::resize({w, h});
    }
};

} // namespace core::cuda