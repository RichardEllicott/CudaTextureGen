/*

cuda stream wrapper

🧩 Usage Example:

core::cuda::Stream stream;  // stream is automatically created
my_kernel<<<grid, block, 0, stream.get()>>>(...); // run a kernel
stream.sync(); // optional: wait for completion

// stream is automatically destroyed when it goes out of scope



⚠️ we still don't use streams for allocation de-allocation
maybe we will develop a stream usage pattern where we use a smartpointer to track an object like this




*/
#pragma once
// #include "types.h"
#include <stdexcept>

// SmartStream
#include <cuda_runtime.h> // For cudaStream_t, cudaStreamCreateWithFlags, cudaStreamDestroy, etc.
#include <memory>         // For std::shared_ptr
#include <stdexcept>      // For std::runtime_error

namespace core::cuda {

// OLD pattern, had copy and move simply banned... this was okay but the smart pointer pattern alows copy and move
/*

class Stream {
    cudaStream_t stream;

  public:
    explicit Stream(unsigned int flags = cudaStreamDefault) {
        cudaError_t err = cudaStreamCreateWithFlags(&stream, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }

    ~Stream() {
        cudaStreamDestroy(stream);
    }

    // Non-copyable
    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;

    //
    //
    //
    // // Non-movable
    // Stream(Stream &&) = delete;
    // Stream &operator=(Stream &&) = delete;

    // Movable
    Stream(Stream &&other) noexcept : stream(other.stream) {
        other.stream = nullptr;
    }
    Stream &operator=(Stream &&other) noexcept {
        if (this != &other) {
            if (stream)
                cudaStreamDestroy(stream);
            stream = other.stream;
            other.stream = nullptr;
        }
        return *this;
    }
    bool valid() const { return stream != nullptr; }

    //
    //
    //

    // Accessor
    cudaStream_t get() const { return stream; }

    void sync() const {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Stream sync failed");
        }
    }
};

*/

// NEW PATTERN using smart pointer, the idea is we can create ones of these
// but we can also copy it, the actual stream will be deleted only when all copies are deleted
struct StreamData {
    cudaStream_t stream;

    StreamData(unsigned int flags = cudaStreamDefault) {
        cudaError_t err = cudaStreamCreateWithFlags(&stream, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }

    ~StreamData() {
        printf("[StreamData] Destroying CUDA stream: %p\n", (void *)stream);

        cudaStreamDestroy(stream);
    }
};

class Stream {
    std::shared_ptr<StreamData> handle;

  public:
    Stream() : handle(std::make_shared<StreamData>()) {}

    explicit Stream(unsigned int flags) : handle(std::make_shared<StreamData>(flags)) {}

    // Copyable and assignable by default
    // Stream(const Stream &) = default;
    // Stream &operator=(const Stream &) = default;

    // these two are the same as hthe defaults above, just add debug messages
    Stream(const Stream &other) : handle(other.handle) {
        printf("[Stream] Copied (ref count = %ld)\n", handle.use_count());
    }

    Stream &operator=(const Stream &other) {
        if (this != &other) {
            handle = other.handle;
            printf("[Stream] Assigned (ref count = %ld)\n", handle.use_count());
        }
        return *this;
    }

    //
    //
    //

    // Accessor
    cudaStream_t get() const {
        return handle ? handle->stream : nullptr;
    }

    void sync() const {
        if (handle) {
            cudaError_t err = cudaStreamSynchronize(handle->stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("Stream sync failed");
            }
        }
    }

    bool valid() const { return handle != nullptr; }
};

} // namespace core::cuda
