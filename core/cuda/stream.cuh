/*

cuda stream wrapper

🧩 Usage Example:

core::cuda::Stream stream;  // stream is automatically created
my_kernel<<<grid, block, 0, stream.get()>>>(...); // run a kernel
stream.sync(); // optional: wait for completion

// stream is automatically destroyed when it goes out of scope

*/
#pragma once
// #include "types.h"
#include <stdexcept>

namespace core::cuda {

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

    // Non-movable
    Stream(Stream &&) = delete;
    Stream &operator=(Stream &&) = delete;

    // Accessor
    cudaStream_t get() const { return stream; }

    void sync() const {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Stream sync failed");
        }
    }
};

} // namespace core