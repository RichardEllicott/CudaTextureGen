/*

*/
#pragma once
// #include "types.h"

namespace core {

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

} // namespace core