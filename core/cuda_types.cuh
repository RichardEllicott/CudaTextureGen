/*

custom Cuda objects, designed to automaticly allocate and free memory, download etc

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

    // // Return the ptr, read only
    // T *device_ptr() const {
    //     return _device_ptr;
    // }

    // Return the ptr, read only
    T *dev_ptr() const {
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

    // // device pointer accessor
    // T *device_ptr() const {
    //     return _device_ptr;
    // }

    // device pointer accessor
    T *dev_ptr() const {
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
    cudaStream_t stream;

  public:
    explicit CudaStream(unsigned int flags = cudaStreamDefault) {
        cudaError_t err = cudaStreamCreateWithFlags(&stream, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }

    ~CudaStream() {
        cudaStreamDestroy(stream);
    }

    // Non-copyable
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    // Non-movable
    CudaStream(CudaStream &&) = delete;
    CudaStream &operator=(CudaStream &&) = delete;

    // Accessor
    cudaStream_t get() const { return stream; }

    void sync() const {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Stream sync failed");
        }
    }
};

template <typename T>
class CudaArrayManager {
  private:
    T *_dev_ptr = nullptr;
    size_t _size = 0;       // number of elements
    size_t _size_bytes = 0; // cached byte size

  public:
    // disallow copy
    CudaArrayManager(const CudaArrayManager &) = delete;
    CudaArrayManager &operator=(const CudaArrayManager &) = delete;

    // allow move
    CudaArrayManager(CudaArrayManager &&other) noexcept {
        *this = std::move(other);
    }
    CudaArrayManager &operator=(CudaArrayManager &&other) noexcept {
        if (this != &other) {
            free_device();
            _dev_ptr = other._dev_ptr;
            _size = other._size;
            _size_bytes = other._size_bytes;
            other._dev_ptr = nullptr;
            other._size = 0;
            other._size_bytes = 0;
        }
        return *this;
    }

    CudaArrayManager() = default;
    ~CudaArrayManager() { free_device(); }

    size_t size() const { return _size; }
    bool empty() const { return _dev_ptr == nullptr; }
    T *dev_ptr() { return _dev_ptr; }
    const T *dev_ptr() const { return _dev_ptr; }

    void resize(size_t n) {
        _size = n;
        allocate_device();
    }

    void allocate_device() {
        if (_size == 0) {
            // nothing to allocate
            free_device();
            return;
        }

        size_t new_size_bytes = _size * sizeof(T);

        // if already allocated with correct size, do nothing
        if (_dev_ptr && _size_bytes == new_size_bytes) {
            return;
        }

        // otherwise, free and reallocate
        free_device();

        _size_bytes = new_size_bytes;
        cudaError_t err = cudaMalloc(&_dev_ptr, _size_bytes);
        if (err != cudaSuccess) {
            _dev_ptr = nullptr;
            _size = 0;
            _size_bytes = 0;
            throw std::runtime_error("cudaMalloc failed");
        }
    }

    void zero_device() {
        if (_size == 0) {
            throw std::runtime_error("zero_device called with size == 0");
        }

        // allocate if not already allocated
        if (!_dev_ptr) {
            allocate_device();
            if (!_dev_ptr) {
                throw std::runtime_error("zero_device failed: allocation failed");
            }
        }

        cudaError_t err = cudaMemset(_dev_ptr, 0, _size_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed");
        }
    }

    void free_device() {
        if (_dev_ptr) {
            cudaError_t err = cudaFree(_dev_ptr);
            if (err != cudaSuccess) {
                // optional: log error
            }
            _dev_ptr = nullptr;
            _size = 0;
            _size_bytes = 0;
        }
    }

    // -------------------------------
    // Host <-> Device transfer methods
    // -------------------------------

    // Upload from host array to device
    void upload(const T *host_ptr, size_t count) {
        if (count > _size) {
            throw std::runtime_error("Upload size exceeds device allocation");
        }
        cudaError_t err = cudaMemcpy(_dev_ptr, host_ptr,
                                     count * sizeof(T),
                                     cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (Host->Device) failed");
        }
    }

    // Download from device to host array
    void download(T *host_ptr, size_t count) const {
        if (count > _size) {
            throw std::runtime_error("Download size exceeds device allocation");
        }
        cudaError_t err = cudaMemcpy(host_ptr, _dev_ptr,
                                     count * sizeof(T),
                                     cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (Device->Host) failed");
        }
    }
};

} // namespace core