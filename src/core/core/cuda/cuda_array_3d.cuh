/*

Cuda version of Array2D, allows automatic device allocation


*/
#pragma once
#include "core/types/array_3d.h"
#include <cuda_runtime.h>
#include <utility> // for std::move

namespace core::cuda {

template <typename T>
class CudaArray3D : public core::types::Array3D<T> {
  private:
    T *_device_ptr = nullptr;
    size_t _device_width = 0;
    size_t _device_height = 0;
    size_t _device_depth = 0;
    size_t _device_allocated_bytes = 0;

    bool _dims_match() const {
        return this->get_width() == _device_width &&
               this->get_height() == _device_height &&
               this->get_depth() == _device_depth;
    }

  public:
    CudaArray3D() = default;

    // Conversion constructor from host Array3D
    explicit CudaArray3D(const core::types::Array3D<T> &src) {
        this->resize(src.get_width(), src.get_height(), src.get_depth());
        std::copy(src.data(), src.data() + src.size(), this->data());
    }

    // Move constructor
    CudaArray3D(CudaArray3D &&other) noexcept
        : core::types::Array3D<T>(std::move(other)),
          _device_ptr(other._device_ptr),
          _device_width(other._device_width),
          _device_height(other._device_height),
          _device_depth(other._device_depth),
          _device_allocated_bytes(other._device_allocated_bytes) {
        other._device_ptr = nullptr;
        other._device_width = other._device_height = other._device_depth = 0;
        other._device_allocated_bytes = 0;
    }

    // Move assignment
    CudaArray3D &operator=(CudaArray3D &&other) noexcept {
        if (this != &other) {
            free_device();
            core::types::Array3D<T>::operator=(std::move(other));
            _device_ptr = other._device_ptr;
            _device_width = other._device_width;
            _device_height = other._device_height;
            _device_depth = other._device_depth;
            _device_allocated_bytes = other._device_allocated_bytes;

            other._device_ptr = nullptr;
            other._device_width = other._device_height = other._device_depth = 0;
            other._device_allocated_bytes = 0;
        }
        return *this;
    }

    // Delete copy operations
    CudaArray3D(const CudaArray3D &) = delete;
    CudaArray3D &operator=(const CudaArray3D &) = delete;

    // Device pointer accessor
    T *dev_ptr() const { return _device_ptr; }

    // Allocate device memory
    void allocate_device() {
        if (this->empty()) {
            free_device();
        } else if (!_dims_match()) {
            free_device();
            _device_width = this->get_width();
            _device_height = this->get_height();
            _device_depth = this->get_depth();
            _device_allocated_bytes = this->size() * sizeof(T);
            cudaMalloc(&_device_ptr, _device_allocated_bytes);
        }
    }

    bool is_allocated() const { return _device_ptr != nullptr; }

    // Upload host → device
    void upload() {
        allocate_device();
        if (!this->empty()) {
            cudaMemcpy(_device_ptr, this->data(),
                       _device_allocated_bytes, cudaMemcpyHostToDevice);
        }
    }

    // Download device → host
    void download() {
        if (_device_ptr) {
            if (!_dims_match()) {
                this->resize(_device_width, _device_height, _device_depth);
            }
            cudaMemcpy(this->data(), _device_ptr, _device_allocated_bytes, cudaMemcpyDeviceToHost);
        }
    }

    // Free device memory
    void free_device() {
        if (_device_ptr) {
            cudaFree(_device_ptr);
            _device_ptr = nullptr;
            _device_width = _device_height = _device_depth = 0;
            _device_allocated_bytes = 0;
        }
    }

    ~CudaArray3D() {
        free_device();
    }
};

} // namespace core::cuda