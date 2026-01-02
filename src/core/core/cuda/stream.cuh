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

// Stream pool
#include <stdexcept>
#include <unordered_map>

namespace core::cuda {

// Stream helper, wil create a stream automaticly
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

class StreamPool {
    std::unordered_map<int, Stream> streams;

  public:
    // Get a stream by index, allocating if necessary
    Stream &get(int id, unsigned int flags = cudaStreamDefault) {
        auto it = streams.find(id);
        if (it != streams.end()) {
            return it->second; // existing Stream
        }
        // lazily emplace a new Stream object
        auto [pos, inserted] = streams.emplace(id, Stream(flags));
        return pos->second;
    }

    // shortcut to get pointer directly
    cudaStream_t get_ptr(int id, unsigned int flags = cudaStreamDefault) {
        return get(id, flags).get();
    }

    // Synchronize all streams
    void sync_all() {
        for (auto &kv : streams) {
            kv.second.sync();
        }
    }

    // Check if a stream exists
    bool has(int id) const {
        return streams.find(id) != streams.end();
    }

    // Remove a stream (its destructor will destroy the CUDA stream)
    void remove(int id) {
        streams.erase(id);
    }

    // Number of streams currently allocated
    size_t size() const { return streams.size(); }
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




} // namespace core::cuda
