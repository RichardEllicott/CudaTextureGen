/*

DeviceArray

may be a more generic template


will have a common interface of "DeviceArrayBase"



*/
#pragma once

#include <array>
#include <cstddef>
#include <cstring>        // memcpy
#include <cuda_runtime.h> // cudaMalloc, cudaFree, cudaMemcpy, cudaMemset, cudaError_t
#include <functional>     // for std::function
#include <iostream>       // if you’re printing/logging inside the callback
#include <memory>         // for std::unique_ptr
#include <stdexcept>      // std::runtime_error
#include <string>         // debug printing
#include <thread>         // for std::thread
#include <utility>        // std::swap (needed for your swap implementation)
#include <vector>         // if you use temporary host buffers

namespace core::cuda {

// BREAKS LINUX???
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
    T *data() { return _ptr.get(); }
    const T *data() const { return _ptr.get(); }
    std::size_t size() const { return _size; }

    // convenience: copy from host
    void copy_from(const T *src, std::size_t n) {
        ensure(n);
        std::memcpy(_ptr.get(), src, n * sizeof(T));
    }

    // free explicitly
    void reset() {
        _ptr.reset();
        _size = 0;
    }

    // non-copyable, but movable
    TempBuffer(const TempBuffer &) = delete;
    TempBuffer &operator=(const TempBuffer &) = delete;
    TempBuffer(TempBuffer &&) noexcept = default;
    TempBuffer &operator=(TempBuffer &&) noexcept = default;
};

class DeviceArrayBase {
  protected:
    // optional stream
    cudaStream_t _stream{nullptr};

    // debug messages
    bool _debug = false;
    std::string _label;

  public:
    bool get_debug() const noexcept { return _debug; } // putting noexcept for style, compiler likely finds these anyway (not sure on this yet i might omit them)
    void set_debug(bool debug) noexcept { _debug = debug; }

    const std::string &get_label() const noexcept { return _label; } // get label marked noexcept, compiler prob done this anyway
    void set_label(std::string label) { _label = std::move(label); }

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
    // deconstructor
    virtual ~DeviceArrayBase() = default;
    // callable on any base object, designed for easy macro usage
    virtual void resize_helper(size_t width, size_t height = 1, size_t depth = 1) = 0;
};

template <typename T, int Dim>
class DeviceArray : public core::cuda::DeviceArrayBase {
    static_assert(Dim > 0, "DeviceArray requires Dim > 0");

    std::array<size_t, Dim> _shape{}; // default dimensions will be 0

    // device side pointer
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

#pragma region DEBUG_MESSAGES

    // for debug
    std::string size_bytes_string() const {
        return std::to_string(size_bytes()) + " bytes";
    }

    // string of shape like 16x16x3 (for debug)
    std::string shape_string() const {
        std::string s;
        s.reserve(32);
        for (size_t i = 0; i < Dim; ++i) {
            s += std::to_string(_shape[i]);
            if (i + 1 < Dim) s += "x";
        }
        return s;
    }

    // print debug data if in debug mode
    void debug_print_upload(const void *host_ptr) const {
        if (!_debug) return;

        printf(
            "upload(): host=%p -> device=%p | shape=%s | bytes=%zu | stream=%p\n",
            host_ptr,
            _dev_ptr,
            shape_string().c_str(),
            size_bytes(),
            (void *)_stream);
    }

#pragma endregion

  public:
    DeviceArray() noexcept {
    }

    ~DeviceArray() override {
        free_device();
    }

    // the total array size
    size_t size() const override {
        size_t total = 1;
        for (auto d : _shape) total *= d;
        return total;
    }

    // array size in bytes
    size_t size_bytes() const override {
        return sizeof(T) * size();
    }

    // shape or dimensions of array
    const std::array<size_t, Dim> &shape() const noexcept {
        return _shape;
    }

    // width always exists
    size_t width() const noexcept {
        return _shape[0];
    }

    // height only if Dim >= 2, else 1
    size_t height() const noexcept {
        if constexpr (Dim >= 2) {
            return _shape[1];
        }
        return 1;
    }

    // depth only if Dim >= 3, else 1
    size_t depth() const noexcept {
        if constexpr (Dim >= 3) {
            return _shape[2];
        }
        return 1;
    }

    // 🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪
    // // Return dimensions in NumPy order (height, width, depth...)
    // std::array<size_t, Dim> numpy_dimensions() const noexcept {
    //     std::array<size_t, Dim> np_dims{};
    //     if constexpr (Dim == 1) {
    //         np_dims[0] = _shape[0]; // same
    //     } else if constexpr (Dim == 2) {
    //         np_dims[0] = _shape[1]; // height
    //         np_dims[1] = _shape[0]; // width
    //     } else if constexpr (Dim == 3) {
    //         np_dims[0] = _shape[1]; // height
    //         np_dims[1] = _shape[0]; // width
    //         np_dims[2] = _shape[2]; // depth
    //     } else {
    //         // For higher dimensions, you can decide a consistent policy
    //         // e.g. swap first two, leave the rest
    //         np_dims = _shape;
    //         std::swap(np_dims[0], np_dims[1]);
    //     }
    //     return np_dims;
    // }
    // 🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪

    // free the device, will also set the dimensions to 0 (which is the same as freeing the device)
    void free_device() override {
        if (_dev_ptr) {
            auto err = cudaFreeAsync(_dev_ptr, _stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaFreeAsync failed");
            }
            _dev_ptr = nullptr;
            _shape = {};
        }
    }

    // device side pointer accessors
    T *dev_ptr() {
        return _dev_ptr;
    }

    // ================================================================================================================================
    // [Resize]
    // --------------------------------------------------------------------------------------------------------------------------------

    // resize dimensions, free and allocate memory as required
    void resize(std::array<size_t, Dim> dimensions) {
        if (_shape == dimensions) {
            return; // nothing changed, skip reallocation
        }

        free_device();
        _shape = dimensions;
        allocate_device();
    }

    // allow accepting any collection of numbers with correct length
    // convert to std::array<size_t, Dim>
    template <typename Container>
    void resize(const Container &c) {

        static_assert(
            std::is_arithmetic<typename Container::value_type>::value,
            "resize(Container) requires arithmetic element type");

        if (c.size() != Dim)
            throw std::runtime_error("resize(Container) requires exactly Dim elements");

        std::array<size_t, Dim> dims{};
        for (size_t i = 0; i < Dim; ++i)
            dims[i] = static_cast<size_t>(c[i]);

        resize(dims); // forward to canonical version
    }

    // resize overload, allows resize(w, h), resize(w, h, d) ...
    template <typename... Sizes>
    void resize(Sizes... sizes) {
        static_assert(sizeof...(Sizes) == Dim, "resize requires exactly Dim arguments");
        resize(std::array<size_t, Dim>{static_cast<size_t>(sizes)...}); // Pack into std::array and forward to canonical resize
    }

    // resize helper, allows resizing 1D, 2D and 3D arrays with the same function, designed for macros
    void resize_helper(size_t width, size_t height = 1, size_t depth = 1) override {
        if constexpr (Dim == 1) {
            if (height != 1 || depth != 1)
                throw std::runtime_error("1D array: height/depth must remain 1");
            resize(std::array<size_t, 1>{width});

        } else if constexpr (Dim == 2) {
            if (depth != 1)
                throw std::runtime_error("2D array: depth must remain 1");
            resize(std::array<size_t, 2>{width, height});

        } else if constexpr (Dim == 3) {
            // all parameters valid
            resize(std::array<size_t, 3>{width, height, depth});

        } else {
            throw std::runtime_error("resize_helper only supports Dim = 1, 2, or 3");
        }
    }

    // ================================================================================================================================

    // fill out device memory with zeros
    void zero_device() override {

        if (!_dev_ptr) return; // no allocated memory (we just pass with no error for now)

        if (_debug) { printf("zero device()...\n"); }

        cudaError_t err = cudaMemsetAsync(_dev_ptr, 0, size_bytes(), _stream);
        if (err != cudaSuccess) throw std::runtime_error("cudaMemset failed");
    }

    // overload with dimensions
    void zero_device(std::array<size_t, Dim> dimensions) {
        resize(dimensions);
        zero_device();
    }

    // wait for sync on this stream (if stream)
    void sync() const {
        if (_stream)
            cudaStreamSynchronize(_stream); // Wait only for this stream
        else
            cudaDeviceSynchronize(); // No stream set → default stream, so wait for all outstanding work
    }

    // Upload from host pointer (optional callback)
    void upload(const T *host_ptr, std::array<size_t, Dim> dimensions, std::function<void()> callback = {}) {

        resize(dimensions); // resize, will reallocate if the size changes
        if (size() == 0) return;

        debug_print_upload(host_ptr);

        cudaError_t err = cudaMemcpyAsync(_dev_ptr, host_ptr, size_bytes(), cudaMemcpyHostToDevice, _stream);
        if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy (Host->Device) failed");

#define CODE_ROUTE 0
#if CODE_ROUTE == 0 // CUDA callback

        if (callback) {
            // Attach host callback
            cudaLaunchHostFunc(
                _stream,
                [](void *userData) {
                    auto *cb = static_cast<std::function<void()> *>(userData);
                    (*cb)();
                    delete cb;
                },
                new std::function<void()>(callback));
        }
#elif CODE_ROUTE == 1 // C++ thread callback

        if (callback) {
            // Launch a detached thread that waits for completion
            std::thread([this, callback]() {
                cudaError_t syncErr = cudaStreamSynchronize(_stream);
                if (syncErr != cudaSuccess) {
                    // You could log or throw here, depending on your design
                    std::cerr << "Upload failed during sync\n";
                }
                callback(); // run user callback
            }).detach();
        }
#endif
#undef CODE_ROUTE
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

#pragma region SWAP
    // swap member function
    void swap(DeviceArray<T, Dim> &other) noexcept {
        using std::swap;
        swap(_dev_ptr, other._dev_ptr);
        swap(_stream, other._stream);
        swap(_shape, other._shape);
    }

    // freind is a syntatic sugar to avoid putting the non-member function outside
    // this allows std::swap(myArrayA, myArrayB) to work
    friend void swap(DeviceArray<T, Dim> &a, DeviceArray<T, Dim> &b) noexcept {
        a.swap(b);
    }
#pragma endregion

#pragma region COPY

    // COPY
    DeviceArray(const DeviceArray &other) {
        _shape = other._shape; // gets the other's shape, now the size_bytes will be correct
        _stream = other._stream;
        _dev_ptr = nullptr;

        // resize(other._shape); // ALTERNATE?? could skip allocation with this

        if (other._dev_ptr) { // if the other is allocated
            auto err = cudaMallocAsync(&_dev_ptr, size_bytes(), _stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMallocAsync failed in copy ctor");
            }

            // copy to this one, from the other
            err = cudaMemcpyAsync(_dev_ptr, other._dev_ptr, size_bytes(), cudaMemcpyDeviceToDevice, _stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemcpyAsync failed in copy ctor");
            }
        }
    }

    DeviceArray &operator=(const DeviceArray &other) {
        if (this != &other) {
            DeviceArray tmp(other); // copy‑construct
            swap(tmp);              // strong exception safety
        }
        return *this;
    }

#pragma endregion

#pragma region MOVE

    DeviceArray(DeviceArray &&other) noexcept {
        swap(other);
        other._dev_ptr = nullptr;
        other._shape = {};
        other._stream = nullptr;
    }

    DeviceArray &operator=(DeviceArray &&other) noexcept {
        if (this != &other) {
            free_device(); // release current
            swap(other);
            other._dev_ptr = nullptr;
            other._shape = {};
            other._stream = nullptr;
        }
        return *this;
    }

#pragma endregion
};

// thin wrapper for 1D
template <typename T>
class DeviceArray1D : public DeviceArray<T, 1> {
  public:
    using Base = DeviceArray<T, 1>;
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
class DeviceArray2D : public DeviceArray<T, 2> {
  public:
    using Base = DeviceArray<T, 2>;
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

    // size_t width() const { return this->shape()[0]; }
    // size_t height() const { return this->shape()[1]; }
};

// thin wrapper for 3D
template <typename T>
class DeviceArray3D : public DeviceArray<T, 3> {
  public:
    using Base = DeviceArray<T, 3>;
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

    // size_t width() const { return this->shape()[0]; }
    // size_t height() const { return this->shape()[1]; }
    // size_t depth() const { return this->shape()[2]; }
};

} // namespace core::cuda