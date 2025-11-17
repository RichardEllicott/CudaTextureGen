/*

cuda array manager, allocates and manages an array on the device

supports copy/swap/move

copy will create a deep copy on the device if data already uploaded


*/
#pragma once

#include <cstddef>        // size_t
#include <cuda_runtime.h> // cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaError_t
#include <stdexcept>      // std::runtime_error
#include <utility>        // std::swap (needed for your swap implementation)
#include <vector>         // std::vector<T>

#include <iostream>
// #include <memory> // smart pointer

// #include "stream.cuh"

namespace core::cuda {

template <typename T>
class DeviceArray {
  private:
    T *_dev_ptr = nullptr;
    size_t _size = 0;       // number of elements
    size_t _size_bytes = 0; // cached byte size

    // std::weak_ptr<Stream> observer = {}; // use a weak ptr?

    // 🔋 WORKING ON NEW FEATURE

#pragma region STREAM_HOOKS

    cudaStream_t _stream = nullptr; // optional stream

    inline cudaError_t _cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) const {
        return cudaMemcpyAsync(dst, src, count, kind, _stream);
    }

    template <typename U>
    inline cudaError_t _cudaMalloc(U **devPtr, size_t size) const {
#if CUDART_VERSION >= 11020
        return cudaMallocAsync(reinterpret_cast<void **>(devPtr), size * sizeof(U), _stream);
#else
        return cudaMalloc(reinterpret_cast<void **>(devPtr), size * sizeof(U));
#endif
    }

    inline cudaError_t _cudaMemset(void *devPtr, int value, size_t count) const {
        return cudaMemsetAsync(devPtr, value, count, _stream);
    }

    inline cudaError_t _cudaFree(void *devPtr) const {
#if CUDART_VERSION >= 11020
        return cudaFreeAsync(devPtr, _stream);
#else
        return cudaFree(devPtr); // fallback if async not available
#endif
    }

#pragma endregion

  public:
    // get/set optional stream
    cudaStream_t get_stream() const { return _stream; }
    void set_stream(cudaStream_t stream) { _stream = stream; }

    // deep copy
    // if there is memory allocated to the GPU it will be copied quickly inside the GPU
    DeviceArray(const DeviceArray &other) {
        if (other._size > 0) {
            resize(other._size);
            cudaError_t err = _cudaMemcpy(_dev_ptr, other._dev_ptr, _size_bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy (Device->Device) failed in copy ctor");
            }
        }
    }
    // deep copy
    DeviceArray &operator=(const DeviceArray &other) {
        if (this != &other) {
            resize(other._size);
            if (_size > 0) {
                cudaError_t err = _cudaMemcpy(_dev_ptr, other._dev_ptr, _size_bytes, cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error("cudaMemcpy (Device->Device) failed in copy assign");
                }
            }
        }
        return *this;
    }

    // move
    DeviceArray(DeviceArray &&other) noexcept {
        *this = std::move(other);
    }
    // move
    DeviceArray &operator=(DeviceArray &&other) noexcept {
        if (this != &other) {
            free_device();
            _dev_ptr = other._dev_ptr;
            _size = other._size;
            _size_bytes = other._size_bytes;
            _stream = other._stream;

            other._dev_ptr = nullptr;
            other._size = 0;
            other._size_bytes = 0;
            other._stream = nullptr;
        }
        return *this;
    }

    // swap member function
    void swap(DeviceArray &other) noexcept {
        std::swap(_dev_ptr, other._dev_ptr);
        std::swap(_size, other._size);
        std::swap(_size_bytes, other._size_bytes);
        std::swap(_stream, other._stream); // new stream
    }

    // swap, need a Friend free function for ADL
    friend void swap(DeviceArray &a, DeviceArray &b) noexcept {
        a.swap(b);
    }

    DeviceArray() = default;
    ~DeviceArray() { free_device(); }

    size_t size() const { return _size; }
    bool empty() const { return _dev_ptr == nullptr; }
    T *dev_ptr() { return _dev_ptr; }
    const T *dev_ptr() const { return _dev_ptr; }

    // resize the array, if we to a size larger than 0, will assign memory to the GPU
    void resize(size_t n) {
        if (n == _size)
            return;

        size_t new_size_bytes = n * sizeof(T);
        if (_dev_ptr && _size_bytes == new_size_bytes) {
            _size = n; // only update size if reallocation is skipped
            return;
        }

        free_device(); // always free before reallocating
        if (n > 0) {
            cudaError_t err = _cudaMalloc(&_dev_ptr, new_size_bytes);

            if (err != cudaSuccess) {
                _dev_ptr = nullptr;
                _size = 0;
                _size_bytes = 0;
                throw std::runtime_error("cudaMalloc failed");
            }
            _size = n;
            _size_bytes = new_size_bytes;
        } else {
            _size = 0;
            _size_bytes = 0;
        }
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
        cudaError_t err = _cudaMalloc(&_dev_ptr, _size_bytes);
        if (err != cudaSuccess) {
            _dev_ptr = nullptr;
            _size = 0;
            _size_bytes = 0;
            throw std::runtime_error("cudaMalloc failed");
        }
    }

    // set the device to clear zero data (floats as 0.0, structs as empty etc)
    void zero_device(cudaStream_t stream = nullptr) {
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

        cudaError_t err = _cudaMemset(_dev_ptr, 0, _size_bytes);

        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed");
        }
    }

    // free the device, return the size to 0
    void free_device() {
        if (_dev_ptr) {
            cudaError_t err = _cudaFree(_dev_ptr);

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

    // Upload from host pointer
    void upload(const T *host_ptr, size_t count) {

        resize(count); // resize, will reallocate if the size changes
        if (count == 0)
            return;

        cudaError_t err = _cudaMemcpy(_dev_ptr, host_ptr, _size_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMemcpy (Host->Device) failed");
    }

    // Download to host pointer (dangerous if the sizes don't match!)
    void download(T *host_ptr) const {

        if (_size <= 0) // skip if no data
            return;

        cudaError_t err = _cudaMemcpy(host_ptr, _dev_ptr, _size_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (Device->Host) failed");
        }
    }

    // -------------------------------
    // Vector-based Host <-> Device transfer methods
    // -------------------------------

    // // Upload from std::vector<T>
    // void upload(const std::vector<T> &host_vec) {
    //     resize(host_vec.size());
    //     if (host_vec.empty())
    //         return;

    //     cudaError_t err = cudaMemcpy(_dev_ptr, host_vec.data(), _size_bytes, cudaMemcpyHostToDevice);
    //     if (err != cudaSuccess) {
    //         throw std::runtime_error("cudaMemcpy (Host->Device) failed in vector upload");
    //     }
    // }

    // // Download into std::vector<T>
    // void download(std::vector<T> &host_vec) const {
    //     if (_size == 0)
    //         return;

    //     host_vec.resize(_size);
    //     cudaError_t err = cudaMemcpy(host_vec.data(), _dev_ptr,
    //                                  _size_bytes, cudaMemcpyDeviceToHost);
    //     if (err != cudaSuccess) {
    //         throw std::runtime_error("cudaMemcpy (Device->Host) failed in vector download");
    //     }
    // }

    // // Convenience: return a new vector with contents
    // std::vector<T> download() const {
    //     std::vector<T> host_vec;
    //     download(host_vec);
    //     return host_vec;
    // }
};

} // namespace core::cuda