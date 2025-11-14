/*

wrapper that uploads data from a local structure to a device side one

*/
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

namespace core::cuda {

// template to allow easy automatic upload/download to cuda, will clear the memory when it goes free
template <typename T>
class Struct {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Struct requires a trivially copyable type");

    T *_device_ptr = nullptr;
    T _host_data{};

  public:
    Struct() = default;
    explicit Struct(const T &value) : _host_data(value) { upload(); }

    // Copy
    Struct(const Struct &other) : _host_data(other._host_data) { upload(); }
    Struct &operator=(const Struct &other) {
        if (this != &other) {
            free_device();
            _host_data = other._host_data;
            upload();
        }
        return *this;
    }

    // Move
    Struct(Struct &&other) noexcept
        : _host_data(std::move(other._host_data)),
          _device_ptr(other._device_ptr) {
        other._device_ptr = nullptr;
    }
    Struct &operator=(Struct &&other) noexcept {
        if (this != &other) {
            free_device();                            // free device data if we have any
            _host_data = std::move(other._host_data); // results in a copy in this case
            _device_ptr = other._device_ptr;          // take the ptr
            other._device_ptr = nullptr;              // make sure the old one looses it's ptr
        }
        return *this;
    }

    // Swap
    friend void swap(Struct &a, Struct &b) noexcept {
        using std::swap;
        swap(a._host_data, b._host_data);
        swap(a._device_ptr, b._device_ptr);
    }

    // Accessors
    T *dev_ptr() const { return _device_ptr; }
    const T &host() const { return _host_data; }
    T &host() { return _host_data; }

    // Upload the host data to device
    void upload() {
        if (!_device_ptr)
            cudaMalloc(&_device_ptr, sizeof(T));
        cudaMemcpy(_device_ptr, &_host_data, sizeof(T), cudaMemcpyHostToDevice);
    }

    // Overload shortcut to upload data
    void upload(T data) {
        _host_data = data;
        upload();
    }

    // Download the device data to _host_data
    void download() {
        if (!_device_ptr)
            throw std::runtime_error("No device memory allocated");
        cudaMemcpy(&_host_data, _device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    }

    // Free device memory
    void free_device() {
        if (_device_ptr) {
            cudaFree(_device_ptr);
            _device_ptr = nullptr;
        }
    }

    ~Struct() { free_device(); }
};

} // namespace core::cuda