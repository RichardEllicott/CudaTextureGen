/*
CUDA compatibility layer
Provides stand‑ins for CUDA vector types and keywords
when CUDA headers are not included.
*/
#pragma once

#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__

#include <cstdint>

// ---- scalar typedefs -------------------------------------------------------

// CUDA‑style aliases, but guaranteed widths
typedef uint32_t uint;      // instead of unsigned int
typedef uint16_t ushort;    // instead of unsigned short
typedef uint8_t uchar;      // instead of unsigned char
typedef uint64_t ulonglong; // instead of unsigned long long

// ---- vector types ----------------------------------------------------------

struct char2 {
    char x, y;
};
struct char3 {
    char x, y, z;
};
struct char4 {
    char x, y, z, w;
};

struct uchar2 {
    uchar x, y;
};
struct uchar3 {
    uchar x, y, z;
};
struct uchar4 {
    uchar x, y, z, w;
};

struct short2 {
    short x, y;
};
struct short3 {
    short x, y, z;
};
struct short4 {
    short x, y, z, w;
};

struct ushort2 {
    ushort x, y;
};
struct ushort3 {
    ushort x, y, z;
};
struct ushort4 {
    ushort x, y, z, w;
};

struct int2 {
    int x, y;
};
struct int3 {
    int x, y, z;
};
struct int4 {
    int x, y, z, w;
};

struct uint2 {
    uint x, y;
};
struct uint3 {
    uint x, y, z;
};
struct uint4 {
    uint x, y, z, w;
};

struct longlong2 {
    long long x, y;
};
struct longlong3 {
    long long x, y, z;
};
struct longlong4 {
    long long x, y, z, w;
};

struct ulonglong2 {
    ulonglong x, y;
};
struct ulonglong3 {
    ulonglong x, y, z;
};
struct ulonglong4 {
    ulonglong x, y, z, w;
};

struct float2 {
    float x, y;
};
struct float3 {
    float x, y, z;
};
struct float4 {
    float x, y, z, w;
};

struct double2 {
    double x, y;
};
struct double3 {
    double x, y, z;
};
struct double4 {
    double x, y, z, w;
};

#endif // __VECTOR_TYPES_H__

// ---- alignment keyword fallback -------------------------------------------

#ifndef __align__
#define __align__(x) alignas(x)
#endif
