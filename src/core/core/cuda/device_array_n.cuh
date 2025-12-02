/*

DeviceArrayN

may be a more generic template


will have a common interface of "DeviceArrayBase"



*/
#pragma once

// #include <cstddef>        // size_t
#include <cuda_runtime.h> // cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaError_t
#include <stdexcept>      // std::runtime_error
#include <utility>        // std::swap (needed for your swap implementation)
// #include <vector>         // std::vector<T>

// #include <iostream>
#include <array>
#include <cstring> // memcpy
#include <memory>  // for std::unique_ptr

namespace core::cuda {


#include <memory>
#include <cstddef>
#include <cstring>


// temp buffer, optional pattern for storing temp array
template <typename T>
class TempBuffer {
    std::unique_ptr<T[]> _ptr;
    std::size_t _size = 0;

public:
    TempBuffer() = default;

    // allocate or reuse if already big enough
    void ensure(std::size_t n) {
        if (!_ptr || _size < n) {
            _ptr.reset(new T[n]);
            _size = n;
        }
    }

    // accessors
    T* data() { return _ptr.get(); }
    const T* data() const { return _ptr.get(); }
    std::size_t size() const { return _size; }

    // convenience: copy from host
    void copy_from(const T* src, std::size_t n) {
        ensure(n);
        std::memcpy(_ptr.get(), src, n * sizeof(T));
    }

    // free explicitly
    void reset() {
        _ptr.reset();
        _size = 0;
    }

    // non-copyable, but movable
    TempBuffer(const TempBuffer&) = delete;
    TempBuffer& operator=(const TempBuffer&) = delete;
    TempBuffer(TempBuffer&&) noexcept = default;
    TempBuffer& operator=(TempBuffer&&) noexcept = default;
};



class DeviceArrayBase {

  protected:
    cudaStream_t _stream{nullptr}; // optional stream

  public:
    // get stream
    cudaStream_t get_stream() const { return _stream; }
    // set optional stream
    void set_stream(cudaStream_t stream) { _stream = stream; }

    // the total array size
    virtual size_t size() const = 0;
    // array size in bytes
    virtual size_t size_bytes() const = 0;
    // free the device, will also set the dimensions to 0 (which is the same as freeing the device)
    virtual void free_device() = 0;
    // initialize device memory to 0s (only if the size() > 0)
    virtual void zero_device() = 0;
    // is empty
    bool empty() const { return size() == 0; }

    //
    virtual ~DeviceArrayBase() = default;
};

template <typename T, int Dim>
class DeviceArrayN : public core::cuda::DeviceArrayBase {

    std::array<size_t, Dim> _dimensions{}; // default dimensions will be 0

    T *_dev_ptr = nullptr;

    // allocate device is private as this is achieved by resizing for the user
    void allocate_device() {
        free_device();

        if (size() == 0) // skip if size is 0
            return;

        auto err = cudaMallocAsync(&_dev_ptr, size_bytes(), _stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMallocAsync failed");
        }
    }

  public:
    DeviceArrayN() noexcept {
    }

    ~DeviceArrayN() override {
        free_device();
    }

    // the total array size
    size_t size() const override {
        size_t total = 1;
        for (auto d : _dimensions) total *= d;
        return total;
    }

    // array size in bytes
    size_t size_bytes() const override {
        return sizeof(T) * size();
    }

    //
    //
    //
    // safer??

    // // return ref to dimensions (might be a bit unsafe?)
    // const std::array<size_t, Dim> &dimensions() const noexcept {
    //     return _dimensions;
    // }

    std::array<size_t, Dim> dimensions() const noexcept {
        return _dimensions;
    }

    // Return dimensions in NumPy order (height, width, depth...)
    std::array<size_t, Dim> numpy_dimensions() const noexcept {
        std::array<size_t, Dim> np_dims{};
        if constexpr (Dim == 1) {
            np_dims[0] = _dimensions[0]; // same
        } else if constexpr (Dim == 2) {
            np_dims[0] = _dimensions[1]; // height
            np_dims[1] = _dimensions[0]; // width
        } else if constexpr (Dim == 3) {
            np_dims[0] = _dimensions[1]; // height
            np_dims[1] = _dimensions[0]; // width
            np_dims[2] = _dimensions[2]; // depth
        } else {
            // For higher dimensions, you can decide a consistent policy
            // e.g. swap first two, leave the rest
            np_dims = _dimensions;
            std::swap(np_dims[0], np_dims[1]);
        }
        return np_dims;
    }

    //
    //
    //

    // free the device, will also set the dimensions to 0 (which is the same as freeing the device)
    void free_device() override {
        if (_dev_ptr) {
            auto err = cudaFreeAsync(_dev_ptr, _stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaFreeAsync failed");
            }
            _dev_ptr = nullptr;
            _dimensions = {};
        }
    }

    T *dev_ptr() {
        return _dev_ptr;
    }

    void resize(std::array<size_t, Dim> dimensions) {
        if (_dimensions == dimensions) {
            return; // nothing changed, skip reallocation
        }

        free_device();
        _dimensions = dimensions;
        allocate_device();
    }

    void zero_device() override {

        if (!_dev_ptr) // no allocated memory (we just pass with no error for now)
            return;

        cudaError_t err = cudaMemsetAsync(_dev_ptr, 0, size_bytes(), _stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed");
        }
    }

    //
    //
    // 🚧 🚧 🚧 🚧
    // // Generic layout converter between SoA and AoS for 3D arrays
    // // direction = true  → SoA → AoS
    // // direction = false → AoS → SoA
    // template <typename T>
    inline void convert_array_layout(const T *source, T *destination, int width, int height, int channels, bool soa_to_aos) {

        // if constexpr (Dim == 1) {

        // }

        if (channels == 1) { // Just copy, no rearrangement needed
            std::memcpy(destination, source, sizeof(T) * width * height);
            return;
        }
        int plane_size = width * height;
        for (int idx = 0; idx < plane_size; ++idx) {
            for (int c = 0; c < channels; ++c) {
                if (soa_to_aos) {
                    destination[idx * channels + c] = source[c * plane_size + idx];
                } else {
                    destination[c * plane_size + idx] = source[idx * channels + c];
                }
            }
        }
    }
    //
    //
    //
    //

    void sync() const {
        if (_stream) {
            // Wait only for this stream
            cudaStreamSynchronize(_stream);
        } else {
            // No stream set → default stream, so wait for all outstanding work
            cudaDeviceSynchronize();
        }
    }

    //
    //
    //
    std::unique_ptr<T[]> _tmp_buffer; // persistent optional buffer
    size_t _tmp_buffer_size = 0;
    void upload(const T *host_ptr, std::array<size_t, Dim> dimensions, bool convert) {
        resize(dimensions);
        if (size() == 0)
            return;

        // ⚠️ new stuff
        const T *src_ptr = host_ptr;
        if (convert) {
            if constexpr (Dim == 3) {
                _tmp_buffer.reset(new T[size()]);                       // allocate temporary
                std::memcpy(_tmp_buffer.get(), host_ptr, size_bytes()); // make a copy
                convert_layout(host_ptr, _tmp_buffer.get(), _dimensions[0], _dimensions[1], _dimensions[2], true);
                src_ptr = _tmp_buffer.get();
            }
        }

        cudaError_t err = cudaMemcpyAsync(_dev_ptr, src_ptr, size_bytes(), cudaMemcpyHostToDevice, _stream);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMemcpyAsync (Host->Device) failed");
    }

    void download(T *host_ptr, bool convert) const {
        if (!_dev_ptr || size() == 0)
            return;

        // ⚠️ new stuff
        const T *src_ptr = host_ptr;

        auto err = cudaMemcpyAsync(host_ptr, _dev_ptr, size_bytes(), cudaMemcpyDeviceToHost, _stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (Device->Host) failed");
        }

        if (convert) {
            if constexpr (Dim == 3) {
                sync();
                _tmp_buffer.reset(new T[size()]);                       // allocate temporary
                std::memcpy(_tmp_buffer.get(), host_ptr, size_bytes()); // make a copy
                convert_layout(host_ptr, _tmp_buffer.get(), _dimensions[0], _dimensions[1], _dimensions[2], false);
                src_ptr = _tmp_buffer.get();
            }
        }
    }

    //
    //
    //

    //
    // ORGINAL
    //
    // Upload from host pointer
    void upload(const T *host_ptr, std::array<size_t, Dim> dimensions) {

        resize(dimensions); // resize, will reallocate if the size changes
        if (size() == 0)
            return;

        cudaError_t err = cudaMemcpyAsync(_dev_ptr, host_ptr, size_bytes(), cudaMemcpyHostToDevice, _stream);
        if (err != cudaSuccess)
            throw std::runtime_error("cudaMemcpy (Host->Device) failed");
    }

    // Download to host pointer (dangerous if the sizes don't match!)
    void download(T *host_ptr) const {
        if (!_dev_ptr || size() == 0)
            return;

        auto err = cudaMemcpyAsync(host_ptr, _dev_ptr, size_bytes(), cudaMemcpyDeviceToHost, _stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy (Device->Host) failed");
        }
    }
    //
    //
    //
    //

#pragma region SWAP
    // swap member function
    void swap(DeviceArrayN<T, Dim> &other) noexcept {
        using std::swap;
        swap(_dev_ptr, other._dev_ptr);
        swap(_stream, other._stream);
        swap(_dimensions, other._dimensions);
    }

    // freind is a syntatic sugar to avoid putting the non-member function outside
    // this allows std::swap(myArrayA, myArrayB) to work
    friend void swap(DeviceArrayN<T, Dim> &a, DeviceArrayN<T, Dim> &b) noexcept {
        a.swap(b);
    }
#pragma endregion
#pragma region COPY

    // COPY
    DeviceArrayN(const DeviceArrayN &other) {
        _dimensions = other._dimensions;
        _stream = other._stream;
        _dev_ptr = nullptr;

        if (other._dev_ptr) {
            auto err = cudaMallocAsync(&_dev_ptr, size_bytes(), _stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMallocAsync failed in copy ctor");
            }

            // resize(_dimensions); // 🚧 we might reduce logic around here

            err = cudaMemcpyAsync(_dev_ptr, other._dev_ptr, size_bytes(), cudaMemcpyDeviceToDevice, _stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemcpyAsync failed in copy ctor");
            }
        }
    }

    DeviceArrayN &operator=(const DeviceArrayN &other) {
        if (this != &other) {
            DeviceArrayN tmp(other); // copy‑construct
            swap(tmp);               // strong exception safety
        }
        return *this;
    }

#pragma endregion

#pragma region MOVE

    DeviceArrayN(DeviceArrayN &&other) noexcept {
        swap(other);
        other._dev_ptr = nullptr;
        other._dimensions = {};
        other._stream = nullptr;
    }

    DeviceArrayN &operator=(DeviceArrayN &&other) noexcept {
        if (this != &other) {
            free_device(); // release current
            swap(other);
            other._dev_ptr = nullptr;
            other._dimensions = {};
            other._stream = nullptr;
        }
        return *this;
    }

#pragma endregion
};

// thin wrapper for 1D
template <typename T>
class DeviceArray1D : public DeviceArrayN<T, 1> {
  public:
    using Base = DeviceArrayN<T, 1>;
    using Base::Base;   // inherit constructors
    using Base::upload; // keep base overloads visible

    // 1D helpers
    void resize(size_t size) {
        Base::resize({size});
    }

    void upload(const T *host_ptr, size_t size) {
        resize(size);
        Base::upload(host_ptr, {size});
    }
};

// thin wrapper for 2D
template <typename T>
class DeviceArray2D : public DeviceArrayN<T, 2> {
  public:
    using Base = DeviceArrayN<T, 2>;
    using Base::Base;   // inherit constructors
    using Base::upload; // keep base overloads visible

    // 2D helpers
    void resize(size_t width, size_t height) {
        Base::resize({width, height});
    }

    void upload(const T *host_ptr, size_t width, size_t height) {
        resize(width, height);
        Base::upload(host_ptr, {width, height});
    }

    size_t width() const { return this->dimensions()[0]; }
    size_t height() const { return this->dimensions()[1]; }
};

// thin wrapper for 3D
template <typename T>
class DeviceArray3D : public DeviceArrayN<T, 3> {
  public:
    using Base = DeviceArrayN<T, 3>;
    using Base::Base;   // inherit constructors
    using Base::upload; // keep base overloads visible

    // 3D helpers
    void resize(size_t width, size_t height, size_t depth) {
        Base::resize({width, height, depth});
    }

    void upload(const T *host_ptr, size_t width, size_t height, size_t depth) {
        resize(width, height, depth);
        Base::upload(host_ptr, {width, height, depth});
    }

    size_t width() const { return this->dimensions()[0]; }
    size_t height() const { return this->dimensions()[1]; }
    size_t depth() const { return this->dimensions()[2]; }
};

} // namespace core::cuda