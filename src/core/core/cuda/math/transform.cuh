/*

*/
#pragma once

#include "core/cuda/math.cuh"
#include <cuda_runtime.h>

#define D_INLINE __device__ __forceinline__           // device only functions
#define DH_INLINE __device__ __host__ __forceinline__ // device and host functions

namespace core::cuda::math {

// basis matrix for easy rotation
struct Basis3 {

    float3 x; // first axis
    float3 y; // second axis
    float3 z; // third axis

    // Identity basis by default
    DH_INLINE Basis3()
        : x(make_float3(1, 0, 0)),
          y(make_float3(0, 1, 0)),
          z(make_float3(0, 0, 1)) {}

    DH_INLINE Basis3(float3 x_, float3 y_, float3 z_)
        : x(x_), y(y_), z(z_) {}

    // Construct from Euler angles (XYZ rotation order)
    DH_INLINE Basis3(float3 angles) {
        float cx = cosf(angles.x), sx = sinf(angles.x);
        float cy = cosf(angles.y), sy = sinf(angles.y);
        float cz = cosf(angles.z), sz = sinf(angles.z);

        // Rotation matrix rows (basis axes)
        x = make_float3(cy * cz, cy * sz, -sy);
        y = make_float3(sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy);
        z = make_float3(cx * sy * cz + sx * sz, cx * sy * sz - sx * cz, cx * cy);
    }

    // Multiply two bases (compose rotations)
    DH_INLINE Basis3 operator*(const Basis3 &B) const {

        return Basis3(
            make_float3(dot(x, B.x), dot(x, B.y), dot(x, B.z)),
            make_float3(dot(y, B.x), dot(y, B.y), dot(y, B.z)),
            make_float3(dot(z, B.x), dot(z, B.y), dot(z, B.z)));
    }

    // ???? extra bit
    DH_INLINE float3 operator*(const float3 &v) const {

        return make_float3(
            dot(x, v),
            dot(y, v),
            dot(z, v));
    }

    // equality
    DH_INLINE bool operator==(const Basis3 &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    // inequality
    DH_INLINE bool operator!=(const Basis3 &other) const {
        return !(*this == other);
    }

    // gives a inverse matrix, so would undo the existing matrix, transpose swaps rows and columns
    DH_INLINE Basis3 transpose() const {
        return Basis3(
            make_float3(x.x, y.x, z.x),
            make_float3(x.y, y.y, z.y),
            make_float3(x.z, y.z, z.z));
    }

    // alias
    DH_INLINE Basis3 inverse() const {
        return transpose(); // orthonormal basis
    }

    DH_INLINE bool nearly_equal(const Basis3 &other, float eps = 1e-6f) const {

        return abs(x.x - other.x.x) < eps &&
               abs(x.y - other.x.y) < eps &&
               abs(x.z - other.x.z) < eps &&
               abs(y.x - other.y.x) < eps &&
               abs(y.y - other.y.y) < eps &&
               abs(y.z - other.y.z) < eps &&
               abs(z.x - other.z.x) < eps &&
               abs(z.y - other.z.y) < eps &&
               abs(z.z - other.z.z) < eps;
    }

    DH_INLINE static Basis3 identity() {
        return Basis3{
            {1.f, 0.f, 0.f},
            {0.f, 1.f, 0.f},
            {0.f, 0.f, 1.f}};
    }

    DH_INLINE bool is_identity() const {
        return nearly_equal(identity());
    }

    // DH_INLINE explicit operator bool() const {
    //     return !is_identity();
    // }

    // corrects the Basis, they can drift
    DH_INLINE void orthonormalize() {

        // normalise x
        x = normalize(x);

        // make y perpendicular to x
        y = y - x * dot(x, y);
        y = normalize(y);

        // compute z as cross product
        z = cross(x, y);

        // 4. normalise z (optional but recommended)
        z = normalize(z);
    }
};

struct Transform {
    Basis3 basis;    // orientation/scale
    float3 position; // translation

    DH_INLINE Transform() = default;

    DH_INLINE Transform(const Basis3 &b, const float3 &p)
        : basis(b), position(p) {}

    // apply transform to a vector (direction only)
    DH_INLINE float3 apply_vector(const float3 &v) const {
        return basis * v; // assuming you have basis3::operator*(float3)
    }

    // apply transform to a point (direction + translation)
    DH_INLINE float3 apply_point(const float3 &p) const {
        return basis * p + position;
    }
};

} // namespace core::cuda::math