/*

⚠️ not using this yet, kept as notes

convert numpy arrays to and from DeviceArray's

new DeviceArray pattern

*/
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace core::arrays {

// Compute strides for row-major layout
template <int Dim>
std::array<size_t, Dim> compute_strides(const std::array<size_t, Dim> &shape) {
    std::array<size_t, Dim> strides{};
    size_t stride = 1;
    for (int i = Dim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Copy from src to dst with permuted axes
template <typename T, int Dim>
void permute_copy(const T *src, const std::array<size_t, Dim> &src_shape, const std::array<int, Dim> &perm, T *dst) {
    // Compute new shape
    std::array<size_t, Dim> dst_shape{};
    for (int i = 0; i < Dim; ++i)
        dst_shape[i] = src_shape[perm[i]];

    // Compute strides
    auto src_strides = compute_strides<Dim>(src_shape);
    auto dst_strides = compute_strides<Dim>(dst_shape);

    // Total elements
    size_t total = 1;
    for (auto s : src_shape) total *= s;

    // Iterate over all elements
    for (size_t linear = 0; linear < total; ++linear) {
        // Decode src index
        std::array<size_t, Dim> idx{};
        size_t tmp = linear;
        for (int i = 0; i < Dim; ++i) {
            idx[i] = tmp / src_strides[i];
            tmp %= src_strides[i];
        }

        // Remap index
        std::array<size_t, Dim> dst_idx{};
        for (int k = 0; k < Dim; ++k)
            dst_idx[k] = idx[perm[k]];

        // Compute offsets
        size_t src_offset = 0, dst_offset = 0;
        for (int i = 0; i < Dim; ++i) {
            src_offset += idx[i] * src_strides[i];
            dst_offset += dst_idx[i] * dst_strides[i];
        }

        dst[dst_offset] = src[src_offset];
    }
}

// 💡 Example: HWC (64,64,3) → CHW (3,64,64)
// std::array<size_t,3> shape = {64,64,3};
// std::array<int,3> perm = {2,0,1};

// std::vector<float> src(64*64*3);
// std::vector<float> dst(64*64*3);

// permute_copy<float,3>(src.data(), shape, perm, dst.data());

// 💡 One liner
// permute_copy<float,3>(
//     src.data(),
//     std::array{64ul,64ul,3ul},   // deduces std::array<size_t,3>
//     std::array{2,0,1},           // deduces std::array<int,3>
//     dst.data()
// );

// Convenience wrapper: allocate dst vector and call permute_copy
template <typename T, int Dim>
std::vector<T> permute_to_vector(const T *src, const std::array<size_t, Dim> &src_shape, const std::array<int, Dim> &perm) {
    // Compute total size
    size_t total = 1;
    for (auto s : src_shape) total *= s;

    // Allocate destination
    std::vector<T> dst(total);

    // Reuse the existing stride-based copy
    permute_copy<T, Dim>(src, src_shape, perm, dst.data());

    return dst;
}

// // 💡 Example: HWC (64,64,3) → CHW (3,64,64)
// std::array<size_t,3> shape = {64,64,3};
// std::array<int,3> perm = {2,0,1};

// std::vector<float> src(64*64*3);
// // fill src with data...

// auto dst = permute_vector<float,3>(src.data(), shape, perm);
// // dst now contains the permuted copy

} // namespace core::arrays