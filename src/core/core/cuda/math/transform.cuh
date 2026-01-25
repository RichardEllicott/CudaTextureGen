/*

*/
#pragma once

#include <cuda_runtime.h>

#include "core/cuda/math.cuh"
#include "core/defines.h"

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

    // corrects the Basis, they can drift
    DH_INLINE void orthonormalize() {

        x = normalize(x); // normalise x

        y = y - x * dot(x, y); // make y perpendicular to x
        y = normalize(y);

        z = cross(x, y); // compute z as cross product

        z = normalize(z); // normalise z (optional but recommended)
    }
};

struct Transform {
    Basis3 basis;
    float3 position;

    DH_INLINE Transform()
        : Transform(identity()) {}

    DH_INLINE Transform(const Basis3 &b, const float3 &p)
        : basis(b), position(p) {}

    DH_INLINE static Transform identity() {
        return Transform(Basis3::identity(), make_float3(0.0f, 0.0f, 0.0f));
    }

    // apply transform to a vector (direction only)
    DH_INLINE float3 apply_vector(const float3 &v) const {
        return basis * v;
    }

    // apply transform to a point (direction + translation)
    DH_INLINE float3 apply_point(const float3 &p) const {
        return basis * p + position;
    }

    // equality
    DH_INLINE bool operator==(const Transform &other) const {
        return basis == other.basis && position == other.position;
    }

    DH_INLINE bool operator!=(const Transform &other) const {
        return !(*this == other);
    }

    // nearly equal
    DH_INLINE bool nearly_equal(const Transform &other, float eps = 1e-6f) const {
        return basis.nearly_equal(other.basis, eps) &&
               fabs(position.x - other.position.x) < eps &&
               fabs(position.y - other.position.y) < eps &&
               fabs(position.z - other.position.z) < eps;
    }

    // identity check
    DH_INLINE bool is_identity(float eps = 1e-6f) const {
        return basis.nearly_equal(Basis3::identity(), eps) &&
               fabs(position.x) < eps &&
               fabs(position.y) < eps &&
               fabs(position.z) < eps;
    }

    // composition
    DH_INLINE Transform operator*(const Transform &B) const {
        return Transform(
            basis * B.basis,
            basis * B.position + position);
    }

    // inverse
    DH_INLINE Transform inverse() const {
        Basis3 invB = basis.inverse();
        float3 invP = invB * (position * -1.0f);
        return Transform(invB, invP);
    }

    // orthonormalize basis
    DH_INLINE void orthonormalize() {
        basis.orthonormalize();
    }
};

// ⚠️ TESTING
// // possible Vector3 type?
// struct Vector3 {
//     float x, y, z;

//     DH_INLINE Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
//     DH_INLINE Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
//     DH_INLINE Vector3(const float3 &v) : x(v.x), y(v.y), z(v.z) {}

//     // implicit conversion to float3
//     DH_INLINE operator float3() const {
//         return make_float3(x, y, z);
//     }
// };

// ⚠️ TESTING
// or perhaps with auto conversion to/from tuple/list
struct Vector3 {
    float x, y, z;

    // Constructors
    DH_INLINE Vector3() : x(0), y(0), z(0) {}
    DH_INLINE Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // Indexing (Python will use this automatically)
    DH_INLINE float &operator[](size_t i) {
        return i == 0 ? x : (i == 1 ? y : z);
    }

    DH_INLINE const float &operator[](size_t i) const {
        return i == 0 ? x : (i == 1 ? y : z);
    }

    // Optional: equality
    DH_INLINE bool operator==(const Vector3 &o) const {
        return x == o.x && y == o.y && z == o.z;
    }

    DH_INLINE bool operator!=(const Vector3 &o) const {
        return !(*this == o);
    }

    // ================================================================
    // [cuda float3]
    // ----------------------------------------------------------------

    // Construct from float3
    DH_INLINE Vector3(const float3 &v) : x(v.x), y(v.y), z(v.z) {}

    // Implicit conversion to float3
    DH_INLINE operator float3() const {
        return make_float3(x, y, z);
    }

    // ----------------------------------------------------------------

    // multiply by scalar to right (vector * scalar)
    DH_INLINE Vector3 operator*(float s) const {
        return Vector3(x * s, y * s, z * s);
    }

    // multiply by scalar to left (scalar * vector)
    DH_INLINE friend Vector3 operator*(float s, const Vector3 &v) {
        return Vector3(v.x * s, v.y * s, v.z * s);
    }

    // Dot product
    DH_INLINE float dot(const Vector3 &o) const {
        return x * o.x + y * o.y + z * o.z;
    }

    // Cross product
    DH_INLINE Vector3 cross(const Vector3 &o) const {
        return Vector3(
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x);
    }
};

} // namespace core::cuda::math