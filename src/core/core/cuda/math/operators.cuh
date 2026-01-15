/*

cuda type operator support

this file being seperate allows import into cuh headers... where the math.cuh can only be imported into .cu files

*/
#pragma once

// #include <cstdint> // uint32_t
#include <cuda_runtime.h>

#define D_INLINE __device__ __forceinline__
#define DH_INLINE __device__ __host__ __forceinline__

#pragma region EQUALITY_OPERATORS // equality for vector 2D and 3D vector types

// 2-component
#define DEFINE_VEC2_EQ_OPS(TYPE)                              \
    DH_INLINE bool operator==(const TYPE &a, const TYPE &b) { \
        return (a.x == b.x) && (a.y == b.y);                  \
    }                                                         \
    DH_INLINE bool operator!=(const TYPE &a, const TYPE &b) { \
        return (a.x != b.x) || (a.y != b.y);                  \
    }

// 3-component
#define DEFINE_VEC3_EQ_OPS(TYPE)                              \
    DH_INLINE bool operator==(const TYPE &a, const TYPE &b) { \
        return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);  \
    }                                                         \
    DH_INLINE bool operator!=(const TYPE &a, const TYPE &b) { \
        return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);  \
    }

// 4-component
#define DEFINE_VEC4_EQ_OPS(TYPE)                                             \
    DH_INLINE bool operator==(const TYPE &a, const TYPE &b) {                \
        return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w); \
    }                                                                        \
    DH_INLINE bool operator!=(const TYPE &a, const TYPE &b) {                \
        return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w); \
    }

// ints
DEFINE_VEC2_EQ_OPS(int2)
DEFINE_VEC3_EQ_OPS(int3)
DEFINE_VEC4_EQ_OPS(int4)

// floats
DEFINE_VEC2_EQ_OPS(float2)
DEFINE_VEC3_EQ_OPS(float3)
DEFINE_VEC4_EQ_OPS(float4)

// doubles
DEFINE_VEC2_EQ_OPS(double2)
DEFINE_VEC3_EQ_OPS(double3)
DEFINE_VEC4_EQ_OPS(double4)

// unsigned ints
DEFINE_VEC2_EQ_OPS(uint2)
DEFINE_VEC3_EQ_OPS(uint3)
DEFINE_VEC4_EQ_OPS(uint4)

// note register preassure is not reduced with shorts and chars, but the array's can be smaller

// // signed chars (8 bit)
// DEFINE_VEC2_EQ_OPS(char2)
// DEFINE_VEC3_EQ_OPS(char3)
// DEFINE_VEC4_EQ_OPS(char4)

// // unsigned chars (8 bit)
// DEFINE_VEC2_EQ_OPS(uchar2)
// DEFINE_VEC3_EQ_OPS(uchar3)
// DEFINE_VEC4_EQ_OPS(uchar4)

// // signed shorts (16 bit)
// DEFINE_VEC2_EQ_OPS(short2)
// DEFINE_VEC3_EQ_OPS(short3)
// DEFINE_VEC4_EQ_OPS(short4)

// // unsigned shorts (16 bit)
// DEFINE_VEC2_EQ_OPS(ushort2)
// DEFINE_VEC3_EQ_OPS(ushort3)
// DEFINE_VEC4_EQ_OPS(ushort4)

// // signed longs (32‑bit on CUDA)
// DEFINE_VEC2_EQ_OPS(long2)
// DEFINE_VEC3_EQ_OPS(long3)
// DEFINE_VEC4_EQ_OPS(long4)

// // unsigned longs
// DEFINE_VEC2_EQ_OPS(ulong2)
// DEFINE_VEC3_EQ_OPS(ulong3)
// DEFINE_VEC4_EQ_OPS(ulong4)

#pragma endregion

#pragma region ADDITION_SUBTRACTION

// 2-component
#define DEFINE_VEC2_ADD_SUB_OPS(TYPE)                        \
    DH_INLINE TYPE operator+(const TYPE &a, const TYPE &b) { \
        return {a.x + b.x, a.y + b.y};                       \
    }                                                        \
    DH_INLINE TYPE &operator+=(TYPE &a, const TYPE &b) {     \
        a.x += b.x;                                          \
        a.y += b.y;                                          \
        return a;                                            \
    }                                                        \
    DH_INLINE TYPE operator-(const TYPE &a, const TYPE &b) { \
        return {a.x - b.x, a.y - b.y};                       \
    }                                                        \
    DH_INLINE TYPE &operator-=(TYPE &a, const TYPE &b) {     \
        a.x -= b.x;                                          \
        a.y -= b.y;                                          \
        return a;                                            \
    }

// 3-component
#define DEFINE_VEC3_ADD_SUB_OPS(TYPE)                        \
    DH_INLINE TYPE operator+(const TYPE &a, const TYPE &b) { \
        return {a.x + b.x, a.y + b.y, a.z + b.z};            \
    }                                                        \
    DH_INLINE TYPE &operator+=(TYPE &a, const TYPE &b) {     \
        a.x += b.x;                                          \
        a.y += b.y;                                          \
        a.z += b.z;                                          \
        return a;                                            \
    }                                                        \
    DH_INLINE TYPE operator-(const TYPE &a, const TYPE &b) { \
        return {a.x - b.x, a.y - b.y, a.z - b.z};            \
    }                                                        \
    DH_INLINE TYPE &operator-=(TYPE &a, const TYPE &b) {     \
        a.x -= b.x;                                          \
        a.y -= b.y;                                          \
        a.z -= b.z;                                          \
        return a;                                            \
    }

// 4-component
#define DEFINE_VEC4_ADD_SUB_OPS(TYPE)                        \
    DH_INLINE TYPE operator+(const TYPE &a, const TYPE &b) { \
        return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; \
    }                                                        \
    DH_INLINE TYPE &operator+=(TYPE &a, const TYPE &b) {     \
        a.x += b.x;                                          \
        a.y += b.y;                                          \
        a.z += b.z;                                          \
        a.w += b.w;                                          \
        return a;                                            \
    }                                                        \
    DH_INLINE TYPE operator-(const TYPE &a, const TYPE &b) { \
        return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; \
    }                                                        \
    DH_INLINE TYPE &operator-=(TYPE &a, const TYPE &b) {     \
        a.x -= b.x;                                          \
        a.y -= b.y;                                          \
        a.z -= b.z;                                          \
        a.w -= b.w;                                          \
        return a;                                            \
    }

// ints
DEFINE_VEC2_ADD_SUB_OPS(int2)
DEFINE_VEC3_ADD_SUB_OPS(int3)
DEFINE_VEC4_ADD_SUB_OPS(int4)

// floats
DEFINE_VEC2_ADD_SUB_OPS(float2)
DEFINE_VEC3_ADD_SUB_OPS(float3)
DEFINE_VEC4_ADD_SUB_OPS(float4)

// doubles
DEFINE_VEC2_ADD_SUB_OPS(double2)
DEFINE_VEC3_ADD_SUB_OPS(double3)
DEFINE_VEC4_ADD_SUB_OPS(double4)

// unsigned ints (broken??)
// DEFINE_VEC2_ADD_SUB_OPS(uint2)
// DEFINE_VEC3_ADD_SUB_OPS(uint3)
// DEFINE_VEC4_ADD_SUB_OPS(uint4)

#undef DEFINE_VEC2_ADD_SUB_OPS
#undef DEFINE_VEC3_ADD_SUB_OPS
#undef DEFINE_VEC4_ADD_SUB_OPS

#pragma endregion

#pragma region SCALAR_OPERATORS // allow [*=, /=] OR [*, /] for vectors, with a scalar on the right

// 2-component
#define DEFINE_VEC2_SCALAR_OPS(TYPE)                   \
    DH_INLINE TYPE operator*(const TYPE &a, float s) { \
        return {a.x * s, a.y * s};                     \
    }                                                  \
    DH_INLINE TYPE &operator*=(TYPE &a, float s) {     \
        a.x *= s;                                      \
        a.y *= s;                                      \
        return a;                                      \
    }                                                  \
    DH_INLINE TYPE operator/(const TYPE &a, float s) { \
        return {a.x / s, a.y / s};                     \
    }                                                  \
    DH_INLINE TYPE &operator/=(TYPE &a, float s) {     \
        a.x /= s;                                      \
        a.y /= s;                                      \
        return a;                                      \
    }

// 3-component
#define DEFINE_VEC3_SCALAR_OPS(TYPE)                   \
    DH_INLINE TYPE operator*(const TYPE &a, float s) { \
        return {a.x * s, a.y * s, a.z * s};            \
    }                                                  \
    DH_INLINE TYPE &operator*=(TYPE &a, float s) {     \
        a.x *= s;                                      \
        a.y *= s;                                      \
        a.z *= s;                                      \
        return a;                                      \
    }                                                  \
    DH_INLINE TYPE operator/(const TYPE &a, float s) { \
        return {a.x / s, a.y / s, a.z / s};            \
    }                                                  \
    DH_INLINE TYPE &operator/=(TYPE &a, float s) {     \
        a.x /= s;                                      \
        a.y /= s;                                      \
        a.z /= s;                                      \
        return a;                                      \
    }

// 4-component
#define DEFINE_VEC4_SCALAR_OPS(TYPE)                   \
    DH_INLINE TYPE operator*(const TYPE &a, float s) { \
        return {a.x * s, a.y * s, a.z * s, a.w * s};   \
    }                                                  \
    DH_INLINE TYPE &operator*=(TYPE &a, float s) {     \
        a.x *= s;                                      \
        a.y *= s;                                      \
        a.z *= s;                                      \
        a.w *= s;                                      \
        return a;                                      \
    }                                                  \
    DH_INLINE TYPE operator/(const TYPE &a, float s) { \
        return {a.x / s, a.y / s, a.z / s, a.w / s};   \
    }                                                  \
    DH_INLINE TYPE &operator/=(TYPE &a, float s) {     \
        a.x /= s;                                      \
        a.y /= s;                                      \
        a.z /= s;                                      \
        a.w /= s;                                      \
        return a;                                      \
    }

// floats
DEFINE_VEC2_SCALAR_OPS(float2)
DEFINE_VEC3_SCALAR_OPS(float3)
DEFINE_VEC4_SCALAR_OPS(float4)

// doubles
DEFINE_VEC2_SCALAR_OPS(double2)
DEFINE_VEC3_SCALAR_OPS(double3)

// DEFINE_VEC4_SCALAR_OPS(double4)

#undef DEFINE_VEC2_SCALAR_OPS
#undef DEFINE_VEC3_SCALAR_OPS
#undef DEFINE_VEC4_SCALAR_OPS

#pragma endregion

#pragma region UNARY_NEGATION_OPERATOR

// __device__ inline float2 operator-(const float2 &v) {
//     return make_float2(-v.x, -v.y);
// }

#pragma endregion
