/*

cuda stream wrapper

🧩 Usage Example:

CudaStream stream;  // stream is automatically created
my_kernel<<<grid, block, 0, stream.get()>>>(...); // run a kernel
stream.sync(); // optional: wait for completion

// stream is automatically destroyed when it goes out of scope

*/
#pragma once
// #include "types.h"
#include <stdexcept>

namespace core {

class CudaStream {
    cudaStream_t stream;

  public:
    explicit CudaStream(unsigned int flags = cudaStreamDefault) {
        cudaError_t err = cudaStreamCreateWithFlags(&stream, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream");
        }
    }

    ~CudaStream() {
        cudaStreamDestroy(stream);
    }

    // Non-copyable
    CudaStream(const CudaStream &) = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    // Non-movable
    CudaStream(CudaStream &&) = delete;
    CudaStream &operator=(CudaStream &&) = delete;

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