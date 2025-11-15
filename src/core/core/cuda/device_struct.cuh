/*

wrapper that uploads data from a local structure to a device side one

*/
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

namespace core::cuda {

// device side host for a structure
template <typename T>
class DeviceStruct {
    static_assert(std::is_trivially_copyable<T>::value,
                  "DeviceStruct requires a trivially copyable type");

    T *_device_ptr = nullptr;

  public:
    DeviceStruct() = default;

    // Construct directly from host value
    explicit DeviceStruct(const T &value) {
        upload(value);
    }

    // Copy
    DeviceStruct(const DeviceStruct &other) {
        if (other._device_ptr) {
            T temp;
            cudaMemcpy(&temp, other._device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
            upload(temp);
        }
    }
    DeviceStruct &operator=(const DeviceStruct &other) {
        if (this != &other) {
            free_device();
            if (other._device_ptr) {
                T temp;
                cudaMemcpy(&temp, other._device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
                upload(temp);
            }
        }
        return *this;
    }

    // Move
    DeviceStruct(DeviceStruct &&other) noexcept
        : _device_ptr(other._device_ptr) {
        other._device_ptr = nullptr;
    }
    DeviceStruct &operator=(DeviceStruct &&other) noexcept {
        if (this != &other) {
            free_device();
            _device_ptr = other._device_ptr;
            other._device_ptr = nullptr;
        }
        return *this;
    }

    // Swap
    friend void swap(DeviceStruct &a, DeviceStruct &b) noexcept {
        using std::swap;
        swap(a._device_ptr, b._device_ptr);
    }

    // Accessor
    T *dev_ptr() const { return _device_ptr; }

    // Upload host data to device
    void upload(const T &data) {
        if (!_device_ptr)
            cudaMalloc(&_device_ptr, sizeof(T));
        cudaMemcpy(_device_ptr, &data, sizeof(T), cudaMemcpyHostToDevice);
    }

    // Download device data to host
    T download() const {
        if (!_device_ptr)
            throw std::runtime_error("No device memory allocated");
        T temp;
        cudaMemcpy(&temp, _device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
        return temp;
    }

    // Free device memory
    void free_device() {
        if (_device_ptr) {
            cudaFree(_device_ptr);
            _device_ptr = nullptr;
        }
    }

    ~DeviceStruct() { free_device(); }
};

} // namespace core::cuda