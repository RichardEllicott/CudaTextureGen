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

    void upload() {

        free_device(); // ultra safe pattern (otherwise need to track size changes better)
        cudaMalloc(&device_ptr, size_bytes());
        cudaMemcpy(device_ptr, this->data(), size_bytes(), cudaMemcpyHostToDevice);
    }

    void download() {

        if (device_ptr) {
            cudaMemcpy(this->data(), device_ptr, size_bytes(), cudaMemcpyDeviceToHost); // note that linux is more script and needs this-> for inherited
        }
    }

    // free allocated device memory
    void free_device() {
        if (device_ptr) {
            cudaFree(device_ptr);
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

  public:
    T host_data{}; // zero data
    T *device_ptr = nullptr;

    CudaStruct() = default;

    explicit CudaStruct(const T &value) : host_data(value) {
        upload(); // automaticly upload to device
    }

    void upload() {
        if (!device_ptr)
            cudaMalloc(&device_ptr, sizeof(T));
        cudaMemcpy(device_ptr, &host_data, sizeof(T), cudaMemcpyHostToDevice);
    }

    void download() {
        // if (!device_ptr)
        //     throw std::runtime_error("No device memory allocated");
        if (device_ptr)
            cudaMemcpy(&host_data, device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    }

    // T *device() const { return device_ptr; }

    // const T &host() const { return host_data; }
    // T &host() { return host_data; }

    void free_device() {
        if (device_ptr) {
            cudaFree(device_ptr);
            device_ptr = nullptr;
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