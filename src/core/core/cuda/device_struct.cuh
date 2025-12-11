/*

wrapper that uploads data from a local structure to a device side one

*/
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>      // revised
#include <type_traits> // revised

namespace core::cuda {


template <typename T>
class DeviceStruct {
    static_assert(std::is_trivially_copyable<T>::value,
                  "DeviceStruct requires a trivially copyable type");

    T *_device_ptr{nullptr};
    cudaStream_t _stream{nullptr};

    static void checkCuda(cudaError_t err, const char *msg) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
        }
    }

  public:
    // Constructors
    DeviceStruct() = default;

    explicit DeviceStruct(const T &value) {
        upload(value);
    }

    // Copy constructor (allocates and performs D2D copy if source has memory)
    DeviceStruct(const DeviceStruct &other)
        : _stream(other._stream) // mirror stream for consistent behavior
    {
        if (other._device_ptr) {
            checkCuda(cudaMalloc(&_device_ptr, sizeof(T)), "cudaMalloc (copy ctor) failed");
            // Use synchronous D2D to ensure constructed object is fully initialized
            checkCuda(cudaMemcpy(_device_ptr, other._device_ptr, sizeof(T),
                                 cudaMemcpyDeviceToDevice),
                      "cudaMemcpy D2D (copy ctor) failed");
        }
    }

    // Copy assignment
    DeviceStruct &operator=(const DeviceStruct &other) {
        if (this == &other)
            return *this;

        // If source has device memory, ensure we have allocation
        if (other._device_ptr) {
            if (!_device_ptr) {
                checkCuda(cudaMalloc(&_device_ptr, sizeof(T)), "cudaMalloc (copy assign) failed");
            }
            // Synchronous D2D for predictability
            checkCuda(cudaMemcpy(_device_ptr, other._device_ptr, sizeof(T),
                                 cudaMemcpyDeviceToDevice),
                      "cudaMemcpy D2D (copy assign) failed");
        } else {
            // Source empty: free ours
            free_device();
        }

        // Mirror stream (does not claim ownership; just follows source behavior)
        _stream = other._stream;
        return *this;
    }

    // Move constructor
    DeviceStruct(DeviceStruct &&other) noexcept
        : _device_ptr(other._device_ptr), _stream(other._stream) {
        other._device_ptr = nullptr;
        other._stream = nullptr;
    }

    // Move assignment
    DeviceStruct &operator=(DeviceStruct &&other) noexcept {
        if (this == &other)
            return *this;
        free_device();
        _device_ptr = other._device_ptr;
        _stream = other._stream;
        other._device_ptr = nullptr;
        other._stream = nullptr;
        return *this;
    }

    // Swap (friend)
    friend void swap(DeviceStruct &a, DeviceStruct &b) noexcept {
        using std::swap;
        swap(a._device_ptr, b._device_ptr);
        swap(a._stream, b._stream);
    }

    // Destructor
    ~DeviceStruct() { free_device(); }

    // Stream accessors
    cudaStream_t get_stream() const noexcept { return _stream; }
    void set_stream(cudaStream_t stream) noexcept { _stream = stream; }

    // Status
    bool valid() const noexcept { return _device_ptr != nullptr; }

    // Accessors
    T *dev_ptr() noexcept { return _device_ptr; }
    const T *dev_ptr() const noexcept { return _device_ptr; }

    // Upload host data to device (allocates if needed)
    void upload(const T &data) {
        if (!_device_ptr) {
            checkCuda(cudaMalloc(&_device_ptr, sizeof(T)), "cudaMalloc failed");
        }
        if (_stream) {
            checkCuda(cudaMemcpyAsync(_device_ptr, &data, sizeof(T),
                                      cudaMemcpyHostToDevice, _stream),
                      "cudaMemcpyAsync H2D failed");
            // Optional: leave unsynchronized for pipelining; caller can sync externally.
            // For safety, you can expose an explicit sync() method.
        } else {
            checkCuda(cudaMemcpy(_device_ptr, &data, sizeof(T),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy H2D failed");
        }
    }

    // Download device data to host (synchronizes if using async)
    T download() const {
        if (!_device_ptr)
            throw std::runtime_error("No device memory allocated");

        T temp;
        if (_stream) {
            checkCuda(cudaMemcpyAsync(&temp, _device_ptr, sizeof(T),
                                      cudaMemcpyDeviceToHost, _stream),
                      "cudaMemcpyAsync D2H failed");
            checkCuda(cudaStreamSynchronize(_stream), "cudaStreamSynchronize failed");
        } else {
            checkCuda(cudaMemcpy(&temp, _device_ptr, sizeof(T),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy D2H failed");
        }
        return temp;
    }

    // Explicit synchronization helper (optional)
    void sync() const {
        if (_stream) {
            checkCuda(cudaStreamSynchronize(_stream), "cudaStreamSynchronize failed");
        }
    }

    // Free device memory (stream-aware)
    void free_device() {
        if (_device_ptr) {
            if (_stream) {
                // cudaFreeAsync available on CUDA 11.2+; falls back to cudaFree if unavailable
                // If you prefer portability, always use cudaFree.
                checkCuda(cudaFreeAsync(_device_ptr, _stream), "cudaFreeAsync failed");
                // Free is enqueued on the stream; nullptr after enqueue
            } else {
                checkCuda(cudaFree(_device_ptr), "cudaFree failed");
            }
            _device_ptr = nullptr;
        }
    }

    // Reset to empty state (explicit)
    void reset() noexcept {
        // Non-throwing reset: best-effort free; ignores errors
        if (_device_ptr) {
            if (_stream)
                (void)cudaFreeAsync(_device_ptr, _stream);
            else
                (void)cudaFree(_device_ptr);
            _device_ptr = nullptr;
        }
        _stream = nullptr;
    }
};

// wrapper for DeviceStruct that maintains a host side copy, also initializes memory by default
template <typename T>
class SyncedDeviceStruct {
    DeviceStruct<T> dev_struct;
    T host_struct{};

  public:
    SyncedDeviceStruct() { dev_struct.upload(host_struct); }
    explicit SyncedDeviceStruct(const T &value) : host_struct(value) { dev_struct.upload(host_struct); }

    // compiler-generated copy/move/dtor are fine
    void upload() { dev_struct.upload(host_struct); }
    void download() { host_struct = dev_struct.download(); }

    T &host() noexcept { return host_struct; }
    const T &host() const noexcept { return host_struct; }
    T *dev_ptr() noexcept { return dev_struct.dev_ptr(); }
    const T *dev_ptr() const noexcept { return dev_struct.dev_ptr(); }

    // Stream control passthrough
    void set_stream(cudaStream_t s) { dev_struct.set_stream(s); }
    cudaStream_t get_stream() const { return dev_struct.get_stream(); }
};



} // namespace core::cuda