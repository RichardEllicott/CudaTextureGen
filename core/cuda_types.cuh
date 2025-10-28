/*

custom types trying to match Godot's patterns (might not be used due to preferance for flat float arrays)

also special Array2D

*/

#pragma once

// #include <cmath>
// #include <vector>
#include "types.h"
#include <cuda_runtime.h> // NOTE we must have header pollution with this, it SHOULD be required

#include <stdexcept>

namespace core {

// added Cuda features
template <typename T>
class CudaArray2D : public Array2D<T> {

  private:
    T *_device_ptr = nullptr;
    size_t _device_width = 0;
    size_t _device_height = 0;
    size_t _device_allocated_bytes = 0;

    bool _array_dim_match_dev_dim() {
        return this->get_width() == _device_width && this->get_height() == _device_height;
    }

  public:
    CudaArray2D() = default;

    // Conversion constructor (build this class from an Array2D)
    CudaArray2D(const Array2D<T> &src) {
        this->resize(src.get_width(), src.get_height());
        std::copy(src.data(), src.data() + src.size(), this->data());
    }

    // Move Constructor (probabally not needed really)
    // ⚠️ AI generated, might need to double check
    //
    //
    // CudaArray2D<float> a;
    // a.resize(1024, 1024);
    // a.upload();
    // CudaArray2D<float> b = std::move(a); // move constructor
    //
    //
    CudaArray2D(CudaArray2D &&other) noexcept {
        *static_cast<Array2D<T> *>(this) = std::move(other); // move base
        _device_ptr = other._device_ptr;
        _device_width = other._device_width;
        _device_height = other._device_height;
        _device_allocated_bytes = other._device_allocated_bytes;

        other._device_ptr = nullptr;
        other._device_width = 0;
        other._device_height = 0;
        other._device_allocated_bytes = 0;
    }

    // Move Assignment Operator ⚠️ AI gen
    CudaArray2D &operator=(CudaArray2D &&other) noexcept {
        if (this != &other) {
            free_device(); // free current device memory

            *static_cast<Array2D<T> *>(this) = std::move(other); // move base
            _device_ptr = other._device_ptr;
            _device_width = other._device_width;
            _device_height = other._device_height;
            _device_allocated_bytes = other._device_allocated_bytes;

            other._device_ptr = nullptr;
            other._device_width = 0;
            other._device_height = 0;
            other._device_allocated_bytes = 0;
        }
        return *this;
    }

    // Delete Copy Operations (prevents accidental shallow copies of _device_ptr)
    // will give a compile error if we try to copy this
    CudaArray2D(const CudaArray2D &) = delete;
    CudaArray2D &operator=(const CudaArray2D &) = delete;

    // Return the ptr, read only
    T *device_ptr() const {
        return _device_ptr;
    }

    // allocate device memory (if not already allocated at correct size and this array is not empty)
    void allocate_device() {

        if (this->empty()) {
            // if empty, ensure device memory freed
            free_device();

        } else if (!_array_dim_match_dev_dim()) {
            // if the size/width has changed since last allocation, ensure free and allocate
            free_device();
            _device_width = this->get_width();
            _device_height = this->get_height();
            _device_allocated_bytes = this->size() * sizeof(T);
            cudaMalloc(&_device_ptr, _device_allocated_bytes);
        }
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
            if (!_array_dim_match_dev_dim()) {
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
            _device_width = 0;
            _device_height = 0;
            _device_allocated_bytes = 0;
        }
    }

    ~CudaArray2D() {
        free_device();
    }
};

// template to allow easy automatic upload/download to cuda, will clear the memory when it goes free
template <typename T>
class CudaStruct {
    static_assert(std::is_trivially_copyable<T>::value, "CudaStruct requires a trivially copyable type");

    T *_device_ptr = nullptr;

  public:
    T host_data{}; // zero data

    CudaStruct() = default;

    // public get pointer
    T *device_ptr() const {
        return _device_ptr;
    }

    // automaticly upload to device if we create with pars, eg:
    //
    // core::CudaStruct<Parameters> gpu_pars(pars); // automaticly uploads and free
    //
    explicit CudaStruct(const T &value) : host_data(value) {
        upload();
    }

    // Non‑copyable, non‑movable (prevents any memory issues)
    CudaStruct(const CudaStruct &) = delete;
    CudaStruct &operator=(const CudaStruct &) = delete;
    CudaStruct(CudaStruct &&) = delete;
    CudaStruct &operator=(CudaStruct &&) = delete;

    void upload() {
        if (!_device_ptr)
            cudaMalloc(&_device_ptr, sizeof(T));
        cudaMemcpy(_device_ptr, &host_data, sizeof(T), cudaMemcpyHostToDevice);
    }

    void download() {
        // if (!device_ptr)
        //     throw std::runtime_error("No device memory allocated");
        if (_device_ptr)
            cudaMemcpy(&host_data, _device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    }

    // T *device() const { return device_ptr; }

    // const T &host() const { return host_data; }
    // T &host() { return host_data; }

    void free_device() {
        if (_device_ptr) {
            cudaFree(_device_ptr);
            _device_ptr = nullptr;
        }
    }

    ~CudaStruct() {
        free_device();
    }
};

/*
Wrapper to handle the stream with auto freeing

🧩 Usage Example:

    CudaStream stream;  // automatically created

    my_kernel<<<grid, block, 0, stream.get()>>>(...);

    stream.sync();      // optional: wait for completion
    // stream is automatically destroyed when it goes out of scope

*/
class CudaStream {
    cudaStream_t stream = nullptr;

  public:
    CudaStream(unsigned int flags = cudaStreamDefault) {
        cudaError_t err = cudaStreamCreateWithFlags(&stream, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }

    ~CudaStream() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }

    // Non-copyable
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    // Movable
    CudaStream(CudaStream &&other) noexcept : stream(other.stream) {
        other.stream = nullptr;
    }

    CudaStream &operator=(CudaStream &&other) noexcept {
        if (this != &other) {
            if (stream)
                cudaStreamDestroy(stream);
            stream = other.stream;
            other.stream = nullptr;
        }
        return *this;
    }

    // Accessor
    cudaStream_t get() const { return stream; }

    // Convenience
    void sync() const {
        cudaStreamSynchronize(stream);
    }
};

} // namespace core