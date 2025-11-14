/*

Cuda version of Array2D, allows automatic device allocation


*/
#pragma once
#include "core/types/array_2d.h"
#include <cuda_runtime.h>
#include <utility> // for std::move

namespace core::cuda {

// added Cuda features
template <typename T>
class CudaArray2D : public core::types::Array2D<T> {

  private:
    T *_device_ptr = nullptr;
    size_t _device_width = 0;
    size_t _device_height = 0;
    size_t _device_allocated_bytes = 0;

    bool _dims_match() {
        return this->get_width() == _device_width && this->get_height() == _device_height;
    }

  public:
    CudaArray2D() = default;

    // Conversion constructor (build this class from an Array2D)
    CudaArray2D(const core::types::Array2D<T> &src) {
        this->resize(src.get_width(), src.get_height());
        std::copy(src.data(), src.data() + src.size(), this->data());
    }

    // // Move Constructor
    // CudaArray2D(CudaArray2D &&other) noexcept {
    //     *static_cast<core::types::Array2D<T> *>(this) = std::move(other); // move base
    //     _device_ptr = other._device_ptr;
    //     _device_width = other._device_width;
    //     _device_height = other._device_height;
    //     _device_allocated_bytes = other._device_allocated_bytes;

    //     other._device_ptr = nullptr;
    //     other._device_width = 0;
    //     other._device_height = 0;
    //     other._device_allocated_bytes = 0;
    // }

    // Delete Copy Operations (prevents accidental shallow copies of _device_ptr)
    // will give a compile error if we try to copy this
    CudaArray2D(const CudaArray2D &) = delete;
    CudaArray2D &operator=(const CudaArray2D &) = delete;

    // Return the ptr, read only
    T *dev_ptr() const {
        return _device_ptr;
    }

    // allocate device memory (if not already allocated at correct size and this array is not empty)
    void allocate_device() {

        if (this->empty()) {
            // if empty, ensure device memory freed
            free_device();

        } else if (!_dims_match()) {
            // if the size/width has changed since last allocation, ensure free and allocate
            free_device();
            _device_width = this->get_width();
            _device_height = this->get_height();
            _device_allocated_bytes = this->size() * sizeof(T);
            cudaMalloc(&_device_ptr, _device_allocated_bytes);
        }
    }

    bool is_allocated() {
        return _device_ptr != nullptr;
    }

    // upload data to the device (will allocate memory if required)
    void upload() {
        allocate_device();
        if (!this->empty()) {
            cudaMemcpy(_device_ptr, this->data(), _device_allocated_bytes, cudaMemcpyHostToDevice);
        }
    }

    // download the data back from the device to the host
    void download() {
        if (_device_ptr) {
            if (!_dims_match()) {
                this->resize(_device_width, _device_height); // on size missmatch resize our host array
            }
            cudaMemcpy(this->data(), _device_ptr, _device_allocated_bytes, cudaMemcpyDeviceToHost);
        }
    }

    // free allocated device memory
    void free_device() {
        if (_device_ptr) {
            cudaFree(_device_ptr);
            _device_ptr = nullptr;
            _device_width = _device_height = 0;
            _device_allocated_bytes = 0;
        }
    }

    ~CudaArray2D() {
        free_device();
    }

    // swap
    void swap(CudaArray2D &other) noexcept {
        // swap base part
        core::types::Array2D<T>::swap(other);

        // swap device part
        std::swap(_device_ptr, other._device_ptr);
        std::swap(_device_width, other._device_width);
        std::swap(_device_height, other._device_height);
        std::swap(_device_allocated_bytes, other._device_allocated_bytes);
    }

    friend void swap(CudaArray2D &a, CudaArray2D &b) noexcept {
        a.swap(b);
    }

    // move by reusing swap
    CudaArray2D(CudaArray2D &&other) noexcept {
        swap(other);
    }

    CudaArray2D &operator=(CudaArray2D &&other) noexcept {
        swap(other);
        return *this;
    }
};

} // namespace core::cuda