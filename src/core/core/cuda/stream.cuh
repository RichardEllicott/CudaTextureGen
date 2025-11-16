/*

cuda stream wrapper

🧩 Usage Example:

core::cuda::Stream stream;  // stream is automatically created
my_kernel<<<grid, block, 0, stream.get()>>>(...); // run a kernel
stream.sync(); // optional: wait for completion

// stream is automatically destroyed when it goes out of scope



⚠️ THE CURRENT SMART POINTER INSIDE PATTERN I BELIEVE TO BE A BIT CONFUSING


*/
#pragma once
// #include "types.h"
#include <stdexcept>

// SmartStream
#include <cuda_runtime.h> // For cudaStream_t, cudaStreamCreateWithFlags, cudaStreamDestroy, etc.
#include <memory>         // For std::shared_ptr
#include <stdexcept>      // For std::runtime_error

// pointer experiments
#include <cuda_runtime.h> // for CUDA functions
#include <memory>         // for std::unique_ptr
#include <stdexcept>      // for std::runtime_error
#include <type_traits>    // for std::remove_pointer_t

namespace core::cuda {

// class Stream {
//     cudaStream_t stream = nullptr;

//   public:
//     explicit Stream(unsigned int flags = cudaStreamDefault) {
//         cudaError_t err = cudaStreamCreateWithFlags(&stream, flags);
//         if (err != cudaSuccess) {
//             throw std::runtime_error(std::string("Failed to create CUDA stream: ") +
//                                      cudaGetErrorString(err));
//         }
//     }

//     // co coopy
//     Stream(const Stream &) = delete;
//     Stream &operator=(const Stream &) = delete;

//     Stream(Stream &&other) noexcept : stream(other.stream) {
//         other.stream = nullptr;
//     }
//     Stream &operator=(Stream &&other) noexcept {
//         if (this != &other) {
//             if (stream)
//                 cudaStreamDestroy(stream);
//             stream = other.stream;
//             other.stream = nullptr;
//         }
//         return *this;
//     }

//     void swap(Stream &other) noexcept {
//         std::swap(stream, other.stream);
//     }

//     bool valid() const noexcept { return stream != nullptr; }
//     cudaStream_t get() const noexcept { return stream; }

//     void sync() const {
//         cudaError_t err = cudaStreamSynchronize(stream);
//         if (err != cudaSuccess) {
//             throw std::runtime_error(std::string("Stream sync failed: ") +
//                                      cudaGetErrorString(err));
//         }
//     }

//     ~Stream() {
//         if (stream)
//             cudaStreamDestroy(stream);
//     }
// };

class Stream {
    cudaStream_t stream = nullptr; // underlying CUDA stream handle

  public:
    // Constructor: creates a CUDA stream with given flags
    explicit Stream(unsigned int flags = cudaStreamDefault) {
        cudaError_t err = cudaStreamCreateWithFlags(&stream, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to create CUDA stream: ") + cudaGetErrorString(err));
        }
    }

    // Deleted copy constructor: streams cannot be copied
    Stream(const Stream &) = delete;

    // Deleted copy assignment: streams cannot be copied
    Stream &operator=(const Stream &) = delete;

    // Move constructor: transfers ownership of the stream
    Stream(Stream &&other) noexcept : stream(other.stream) {
        other.stream = nullptr;
    }

    // Move assignment: destroys current stream, then takes ownership from other
    Stream &operator=(Stream &&other) noexcept {
        if (this != &other) {
            if (stream)
                cudaStreamDestroy(stream);
            stream = other.stream;
            other.stream = nullptr;
        }
        return *this;
    }

    // Swap function: exchanges stream handles between two Stream objects
    void swap(Stream &other) noexcept {
        std::swap(stream, other.stream);
    }

    // Check if the stream is valid (not null)
    bool valid() const noexcept { return stream != nullptr; }

    // Accessor: returns the raw cudaStream_t handle
    cudaStream_t get() const noexcept { return stream; }

    // Synchronize: blocks until all work in the stream is finished
    void sync() const {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Stream sync failed: ") +
                                     cudaGetErrorString(err));
        }
    }

    // Destructor: destroys the CUDA stream if it exists
    ~Stream() {
        if (stream) {
            // cudaStreamSynchronize(stream); // OPTIONAL wait for all work to finish
            cudaStreamDestroy(stream);
        }
    }
};

#pragma region SHARED_PTR_HOSTING_STREAM_EXAMPLE


// this EXAMPLE is a good idea to allow sharing the ptr
inline void share_with_weak_ptr_example() {

    auto streamA = std::make_shared<Stream>(cudaStreamDefault);

    std::weak_ptr<Stream> observer = streamA; // B observes

    // safe and best pattern, lock would increase the ref count if the pointer still exists
    if (auto s = observer.lock()) {
        s->sync(); // safe to use
    } else {
        // stream no longer exists
    }
}

#pragma endregion

#pragma region UNIQUE_PTR_ALTERNATIVE
//
// EXAMPLE USING A WRAPPED unique_ptr (not really a very good pattern, option to use it internally perhaps)
//

// Custom deleter for CUDA streams.
// - unique_ptr calls this when the StreamHandle goes out of scope.
// - The operator() signature matches what unique_ptr expects: it receives the raw pointer.
// - cudaStream_t is itself a pointer type, so std::remove_pointer_t<cudaStream_t>
//   gives the opaque pointee type. The deleter then receives a cudaStream_t.
struct CudaStreamDeleter {
    void operator()(std::remove_pointer_t<cudaStream_t> *stream) const {
        if (stream) {
            // Destroy the CUDA stream; return value ignored here.
            cudaStreamDestroy(stream);
        }
    }
};

// Define a unique_ptr type that owns a CUDA stream.
// - Element type: the pointee of cudaStream_t (opaque struct).
// - Deleter: CudaStreamDeleter, which calls cudaStreamDestroy.
using StreamHandle = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CudaStreamDeleter>;

// Factory function to create a CUDA stream and wrap it in a StreamHandle.
// - Calls cudaStreamCreateWithFlags to allocate the stream.
// - Throws std::runtime_error if creation fails.
// - Returns a unique_ptr that will automatically destroy the stream when it goes out of scope.
inline StreamHandle create_stream(unsigned int flags = cudaStreamDefault) {
    cudaStream_t s;
    cudaError_t err = cudaStreamCreateWithFlags(&s, flags);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream: " +
                                 std::string(cudaGetErrorString(err)));
    }
    return StreamHandle(s); // unique_ptr takes ownership of the raw handle
}


inline void using_unique_ptr_example() {
    // Create a CUDA stream wrapped in a unique_ptr
    auto stream_handle = create_stream();

    // Get the raw cudaStream_t when you need to pass it to CUDA APIs
    cudaStream_t raw = stream_handle.get();

    // Example: launch a kernel or do a memcpy on this stream
    // my_kernel<<<grid, block, 0, raw>>>(...);

    // Synchronize the stream explicitly if needed
    cudaError_t err = cudaStreamSynchronize(raw);
    if (err != cudaSuccess) {
        throw std::runtime_error("Stream sync failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // When stream_handle goes out of scope, the deleter runs and calls cudaStreamDestroy(raw).
}





#pragma endregion

} // namespace core::cuda
