/*

pure 1D array manager, allocates and manages a standard array on the device

normally i use the Array2D one instead, this one keeps no local copy however

*/
#pragma once
#include <stdexcept>

namespace core::cuda {

template <typename T>
class DeviceArray {
  private:
    T *_dev_ptr = nullptr;
    size_t _size = 0;       // number of elements
    size_t _size_bytes = 0; // cached byte size

  public:
    // disallow copy
    DeviceArray(const DeviceArray &) = delete;
    DeviceArray &operator=(const DeviceArray &) = delete;

    // allow move
    DeviceArray(DeviceArray &&other) noexcept {
        *this = std::move(other);
    }
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

    // Explicit deep copy via clone()... could have used the copy sematics but this is more explicit
    DeviceArray clone() const {
        DeviceArray copy;
        if (_size > 0) {
            copy.resize(_size);
            cudaError_t err = cudaMemcpy(copy._dev_ptr, _dev_ptr,
                                         _size_bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy (Device->Device) failed in clone()");
            }
        }
        return copy;
    }

    DeviceArray() = default;
    ~DeviceArray() { free_device(); }

    size_t size() const { return _size; }
    bool empty() const { return _dev_ptr == nullptr; }
    T *dev_ptr() { return _dev_ptr; }
    const T *dev_ptr() const { return _dev_ptr; }

    void resize(size_t n) {

        if (n == _size) // skip if the same size
            return;

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
};

} // namespace core::cuda