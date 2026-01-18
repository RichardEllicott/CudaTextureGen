/*

cuda type operator support

this file being seperate allows import into cuh headers... where the math.cuh can only be imported into .cu files

*/
#pragma once

#include <cstdint> // uint32_t (required for gcc)
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

#pragma endregion

#pragma region SCALAR_OPERATORS // allow [*=, /=] OR [*, /] for vectors, with a scalar on the right

// 2-component
#define DEFINE_VEC2_SCALAR_OPS(TYPE, SCALAR)            \
    DH_INLINE TYPE operator*(const TYPE &a, SCALAR s) { \
        return {a.x * s, a.y * s};                      \
    }                                                   \
    DH_INLINE TYPE &operator*=(TYPE &a, SCALAR s) {     \
        a.x *= s;                                       \
        a.y *= s;                                       \
        return a;                                       \
    }                                                   \
    DH_INLINE TYPE operator/(const TYPE &a, SCALAR s) { \
        return {a.x / s, a.y / s};                      \
    }                                                   \
    DH_INLINE TYPE &operator/=(TYPE &a, SCALAR s) {     \
        a.x /= s;                                       \
        a.y /= s;                                       \
        return a;                                       \
    }

// 3-component
#define DEFINE_VEC3_SCALAR_OPS(TYPE, SCALAR)            \
    DH_INLINE TYPE operator*(const TYPE &a, SCALAR s) { \
        return {a.x * s, a.y * s, a.z * s};             \
    }                                                   \
    DH_INLINE TYPE &operator*=(TYPE &a, SCALAR s) {     \
        a.x *= s;                                       \
        a.y *= s;                                       \
        a.z *= s;                                       \
        return a;                                       \
    }                                                   \
    DH_INLINE TYPE operator/(const TYPE &a, SCALAR s) { \
        return {a.x / s, a.y / s, a.z / s};             \
    }                                                   \
    DH_INLINE TYPE &operator/=(TYPE &a, SCALAR s) {     \
        a.x /= s;                                       \
        a.y /= s;                                       \
        a.z /= s;                                       \
        return a;                                       \
    }

// 4-component
#define DEFINE_VEC4_SCALAR_OPS(TYPE, SCALAR)            \
    DH_INLINE TYPE operator*(const TYPE &a, SCALAR s) { \
        return {a.x * s, a.y * s, a.z * s, a.w * s};    \
    }                                                   \
    DH_INLINE TYPE &operator*=(TYPE &a, SCALAR s) {     \
        a.x *= s;                                       \
        a.y *= s;                                       \
        a.z *= s;                                       \
        a.w *= s;                                       \
        return a;                                       \
    }                                                   \
    DH_INLINE TYPE operator/(const TYPE &a, SCALAR s) { \
        return {a.x / s, a.y / s, a.z / s, a.w / s};    \
    }                                                   \
    DH_INLINE TYPE &operator/=(TYPE &a, SCALAR s) {     \
        a.x /= s;                                       \
        a.y /= s;                                       \
        a.z /= s;                                       \
        a.w /= s;                                       \
        return a;                                       \
    }

#pragma endregion

#pragma region UNARY_NEGATION_OPERATOR

#define DEFINE_VEC2_UNARY_OPS(TYPE)           \
    DH_INLINE TYPE operator-(const TYPE &a) { \
        return {-a.x, -a.y};                  \
    }                                         \
    DH_INLINE TYPE operator+(const TYPE &a) { \
        return a;                             \
    }

#define DEFINE_VEC3_UNARY_OPS(TYPE)           \
    DH_INLINE TYPE operator-(const TYPE &a) { \
        return {-a.x, -a.y, -a.z};            \
    }                                         \
    DH_INLINE TYPE operator+(const TYPE &a) { \
        return a;                             \
    }

#define DEFINE_VEC4_UNARY_OPS(TYPE)           \
    DH_INLINE TYPE operator-(const TYPE &a) { \
        return {-a.x, -a.y, -a.z, -a.w};      \
    }                                         \
    DH_INLINE TYPE operator+(const TYPE &a) { \
        return a;                             \
    }

#pragma endregion

// ================================================================================================================================
// [Equality]
// --------------------------------------------------------------------------------------------------------------------------------
DEFINE_VEC2_EQ_OPS(int2)
DEFINE_VEC3_EQ_OPS(int3)
DEFINE_VEC4_EQ_OPS(int4)

DEFINE_VEC2_EQ_OPS(float2)
DEFINE_VEC3_EQ_OPS(float3)
DEFINE_VEC4_EQ_OPS(float4)

// DEFINE_VEC2_EQ_OPS(double2)
// DEFINE_VEC3_EQ_OPS(double3)
// DEFINE_VEC4_EQ_OPS(double4)

// DEFINE_VEC2_EQ_OPS(uint2)
// DEFINE_VEC3_EQ_OPS(uint3)
// DEFINE_VEC4_EQ_OPS(uint4)

#undef DEFINE_VEC2_EQ_OPS
#undef DEFINE_VEC3_EQ_OPS
#undef DEFINE_VEC4_EQ_OPS

// ================================================================================================================================
// [Addition and Sbtraction]
// --------------------------------------------------------------------------------------------------------------------------------

DEFINE_VEC2_ADD_SUB_OPS(int2)
DEFINE_VEC3_ADD_SUB_OPS(int3)
DEFINE_VEC4_ADD_SUB_OPS(int4)

DEFINE_VEC2_ADD_SUB_OPS(float2)
DEFINE_VEC3_ADD_SUB_OPS(float3)
DEFINE_VEC4_ADD_SUB_OPS(float4)

// DEFINE_VEC2_ADD_SUB_OPS(double2)
// DEFINE_VEC3_ADD_SUB_OPS(double3)
// DEFINE_VEC4_ADD_SUB_OPS(double4)

DEFINE_VEC2_ADD_SUB_OPS(uint2)
DEFINE_VEC3_ADD_SUB_OPS(uint3)
DEFINE_VEC4_ADD_SUB_OPS(uint4)

#undef DEFINE_VEC2_ADD_SUB_OPS
#undef DEFINE_VEC3_ADD_SUB_OPS
#undef DEFINE_VEC4_ADD_SUB_OPS

// ================================================================================================================================
// [Scalar Ops]
// --------------------------------------------------------------------------------------------------------------------------------

DEFINE_VEC2_SCALAR_OPS(int2, int)
DEFINE_VEC3_SCALAR_OPS(int3, int)
DEFINE_VEC4_SCALAR_OPS(int4, int)

DEFINE_VEC2_SCALAR_OPS(float2, float)
DEFINE_VEC3_SCALAR_OPS(float3, float)
DEFINE_VEC4_SCALAR_OPS(float4, float)

// DEFINE_VEC2_SCALAR_OPS(double2)
// DEFINE_VEC3_SCALAR_OPS(double3)
// DEFINE_VEC4_SCALAR_OPS(double4)

#undef DEFINE_VEC2_SCALAR_OPS
#undef DEFINE_VEC3_SCALAR_OPS
#undef DEFINE_VEC4_SCALAR_OPS

// ================================================================================================================================
// [Unary]
// --------------------------------------------------------------------------------------------------------------------------------

DEFINE_VEC2_UNARY_OPS(int2)
DEFINE_VEC3_UNARY_OPS(int3)
DEFINE_VEC4_UNARY_OPS(int4)

DEFINE_VEC2_UNARY_OPS(float2)
DEFINE_VEC3_UNARY_OPS(float3)
DEFINE_VEC4_UNARY_OPS(float4)

#undef DEFINE_VEC2_UNARY_OPS
#undef DEFINE_VEC3_UNARY_OPS
#undef DEFINE_VEC4_UNARY_OPS

// ================================================================================================================================
// [Undef]
// --------------------------------------------------------------------------------------------------------------------------------
