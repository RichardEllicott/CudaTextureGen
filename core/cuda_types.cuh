/*

custom types trying to match Godot's patterns (might not be used due to preferance for flat float arrays)

also special Array2D

*/

#pragma once

// #include <cmath>
// #include <vector>
#include "types.h"
#include <cuda_runtime.h> // NOTE we must have header pollution with this, it SHOULD be required

namespace core {

// added Cuda features
template <typename T>
class CudaArray2D : public Array2D<T> {

  private:
  public:
    T *device_ptr = nullptr;
    size_t device_width;
    size_t device_height;

    CudaArray2D() = default;

    // Conversion constructor
    CudaArray2D(const Array2D<T> &src) {
        this->resize(src.get_width(), src.get_height());
        std::copy(src.data(), src.data() + src.size(), this->data());
    }

    size_t size_bytes() {
        return this->size() * sizeof(T);
    }

    void upload_to_device() {

        free_device_memory();

        cudaMalloc(&device_ptr, size_bytes());
        cudaMemcpy(device_ptr, this->data(), size_bytes(), cudaMemcpyHostToDevice);
    }

    void download_from_device() {

        if (device_ptr) {
            cudaMemcpy(this->data(), device_ptr, size_bytes(), cudaMemcpyDeviceToHost); // note that linux is more script and needs this-> for inherited
        }
    }

    void free_device_memory() {
        if (device_ptr) {
            cudaFree(device_ptr);
        }
    }

    ~CudaArray2D() {
        free_device_memory();
    }
};

// template to allow easy automatic upload/download to cuda, will clear the memory when it goes free
template <typename T>
class CudaStruct {
    static_assert(std::is_trivially_copyable<T>::value, "CudaStruct requires a trivially copyable type");

  public:
    T host_data{}; // zero data
    T *device_ptr = nullptr;

    CudaStruct() = default;

    explicit CudaStruct(const T &value) : host_data(value) {
        upload_to_device(); // automaticly upload to device
    }

    void upload_to_device() {
        if (!device_ptr)
            cudaMalloc(&device_ptr, sizeof(T));
        cudaMemcpy(device_ptr, &host_data, sizeof(T), cudaMemcpyHostToDevice);
    }

    void download_from_device() {
        // if (!device_ptr)
        //     throw std::runtime_error("No device memory allocated");
        if (device_ptr)
            cudaMemcpy(&host_data, device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    }

    // T *device() const { return device_ptr; }

    // const T &host() const { return host_data; }
    // T &host() { return host_data; }

    void free_device_memory() {
        if (device_ptr) {
            cudaFree(device_ptr);
            device_ptr = nullptr;
        }
    }

    ~CudaStruct() {
        free_device_memory();
    }
};

} // namespace core