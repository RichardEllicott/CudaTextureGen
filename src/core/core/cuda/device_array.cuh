/*

cuda array manager, allocates and manages an array on the device


⚠️ much written by co-pilot, i'm pretty sure it's all good, still slightly learning about the swap/copy/move sematics however


*/
#pragma once

#include <cstddef>        // size_t
#include <cuda_runtime.h> // cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaError_t
#include <stdexcept>      // std::runtime_error
#include <utility>        // std::swap (needed for your swap implementation)
#include <vector>         // std::vector<T>

#include <iostream>
#include <memory> // smart pointer

namespace core::cuda {

template <typename T>
class DeviceArray {
  private:
    T *_dev_ptr = nullptr;
    size_t _size = 0;       // number of elements
    size_t _size_bytes = 0; // cached byte size

  public:
    // deep copy
    // if there is memory allocated to the GPU it will be copied quickly inside the GPU
    DeviceArray(const DeviceArray &other) {
        if (other._size > 0) {
            resize(other._size);
            cudaError_t err = cudaMemcpy(_dev_ptr, other._dev_ptr,
                                         _size_bytes, cudaMemcpyDeviceToDevice);
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
                cudaError_t err = cudaMemcpy(_dev_ptr, other._dev_ptr,
                                             _size_bytes, cudaMemcpyDeviceToDevice);
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
            other._dev_ptr = nullptr;
            other._size = 0;
            other._size_bytes = 0;
        }
        return *this;
    }

    // swap member function
    void swap(DeviceArray &other) noexcept {
        std::swap(_dev_ptr, other._dev_ptr);
        std::swap(_size, other._size);
        std::swap(_size_bytes, other._size_bytes);
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
            cudaError_t err = cudaMalloc(&_dev_ptr, new_size_bytes);
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
        cudaError_t err = cudaMalloc(&_dev_ptr, _size_bytes);
        if (err != cudaSuccess) {
            _dev_ptr = nullptr;
            _size = 0;
            _size_bytes = 0;
            throw std::runtime_error("cudaMalloc failed");
        }
    }

    // set the device to clear zero data (floats as 0.0, structs as empty etc)
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

    // free the device, return the size to 0
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

    // Upload from host pointer
    void upload(const T *host_ptr, size_t count) {

        resize(count); // resize, will reallocate if the size changes
        if (count == 0)
            return;

        cudaError_t err = cudaMemcpy(_dev_ptr, host_ptr, _size_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMemcpy (Host->Device) failed");
    }

    // Download to host pointer (dangerous if the sizes don't match!)
    void download(T *host_ptr) const {

        if (_size <= 0) // skip if no data
            return;

        cudaError_t err = cudaMemcpy(host_ptr, _dev_ptr, _size_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (Device->Host) failed");
        }
    }

    // -------------------------------
    // Vector-based Host <-> Device transfer methods
    // -------------------------------

    // Upload from std::vector<T>
    void upload(const std::vector<T> &host_vec) {
        resize(host_vec.size());
        if (host_vec.empty())
            return;

        cudaError_t err = cudaMemcpy(_dev_ptr, host_vec.data(),
                                     _size_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (Host->Device) failed in vector upload");
        }
    }

    // Download into std::vector<T>
    void download(std::vector<T> &host_vec) const {
        if (_size == 0)
            return;

        host_vec.resize(_size);
        cudaError_t err = cudaMemcpy(host_vec.data(), _dev_ptr,
                                     _size_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (Device->Host) failed in vector download");
        }
    }

    // Convenience: return a new vector with contents
    std::vector<T> download() const {
        std::vector<T> host_vec;
        download(host_vec);
        return host_vec;
    }
};

template <typename T>
class DeviceArray2D : public DeviceArray<T> {
  private:
    size_t _width = 0;
    size_t _height = 0;

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

static void example() {

    int width = 16, height = 16;
    auto device_array = core::cuda::DeviceArray<float>(); // create array
    device_array.resize(width * height);                  // resizing allocates memory
    device_array.zero_device();                           // ensure memory is initialized as 0
}

static void smart_pointer_example() {

    // Create a shared DeviceArray<float> with 10 elements
    auto arr = std::make_shared<DeviceArray<float>>();
    arr->resize(10);

    // Fill a host vector with some values
    std::vector<float> host_data(10);
    for (int i = 0; i < 10; ++i) {
        host_data[i] = static_cast<float>(i) * 0.5f;
    }

    // Upload to device
    arr->upload(host_data);

    // Share the same device array with another smart pointer
    std::shared_ptr<DeviceArray<float>> arr2 = arr;

    // Download from arr2 (same underlying device memory)
    std::vector<float> result;
    arr2->download(result);

    // Print results
    for (float f : result) {
        std::cout << f << " ";
    }
    std::cout << "\n";

    // Both arr and arr2 go out of scope here.
    // Device memory is freed automatically once the last shared_ptr is destroyed.
}

} // namespace core::cuda