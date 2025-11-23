/*

DeviceArrayN

may be a more generic template


will have a common interface of "DeviceArrayNBase"



*/
#pragma once

// #include <cstddef>        // size_t
#include <cuda_runtime.h> // cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaError_t
// #include <stdexcept>      // std::runtime_error
// #include <utility>        // std::swap (needed for your swap implementation)
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

    virtual void free_device() = 0;
    virtual ~DeviceArrayNBase() = default;
};

template <typename T, int Dim>
class DeviceArrayN : public core::cuda::DeviceArrayNBase {

    std::array<size_t, Dim> _dimensions{}; // default dimensions will be 0

    // the total array size
    size_t size() const {
        size_t total = 1;
        for (auto d : _dimensions) total *= d;
        return total;
    }

    // array size in bytes
    size_t size_bytes() const {
        return sizeof(T) * size();
    }

    T *_dev_ptr = nullptr;

  public:
    DeviceArrayN() noexcept {
    }

    void allocate_device() {
        free_device();
        auto err = cudaMallocAsync(&_dev_ptr, size_bytes(), _stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMallocAsync failed");
        }
    }

    void free_device() override {
        if (_dev_ptr) {
            auto err = cudaFreeAsync(_dev_ptr, _stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaFreeAsync failed");
            }
            _dev_ptr = nullptr;
        }
    }

    T *dev_ptr() {
        return _dev_ptr;
    }

    void resize(std::array<size_t, Dim> dimensions) {
        if (_dimensions == dimensions) {
            // nothing changed, skip reallocation
            return;
        }

        free_device();
        _dimensions = dimensions;
        allocate_device();
    }

    void zero_device() {

        if (!dev_ptr) // no allocated memory
            return;


        // concider error, throw std::runtime_error("zero_device called with size == 0");

        cudaError_t err = cudaMemsetAsync(_dev_ptr, 0, size_bytes(), _stream);

        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed");
        }
    }
};

} // namespace core::cuda