// #include "core/cuda/curand_array_2d.cuh"
#include "noise.cuh"
#include <stdexcept> // std::runtime_error

#include "noise_generator.cuh"

namespace TEMPLATE_NAMESPACE {

// Modern integer hash (based on MurmurHash3 finalizer)
__device__ __forceinline__ int hash_int(int x, int y, int z, int seed) {
    int n = x + y * 374761393 + z * 668265263 + seed * 1274126177;

    n ^= n >> 16;
    n *= 0x85ebca6b;
    n ^= n >> 13;
    n *= 0xc2b2ae35;
    n ^= n >> 16;

    return n & 0x7fffffff; // Keep positive for compatibility
}

// dot product for 3D
__device__ __forceinline__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// simple interpolate
__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Creates an S-curve (sigmoid-like shape)
__device__ __forceinline__ float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

// positive modulo wrap (note it might be faster to concider other wrap methods depending on the situation)
__device__ __forceinline__ int posmod(int value, int mod) {
    int result = value % mod;
    return result < 0 ? result + mod : result;
}

// Apply rotation to a point (CUDA device function)
__device__ __forceinline__ float3 rotate(float3 p, const float *r) {
    return make_float3(
        r[0] * p.x + r[1] * p.y + r[2] * p.z,
        r[3] * p.x + r[4] * p.y + r[5] * p.z,
        r[6] * p.x + r[7] * p.y + r[8] * p.z);
}

// 3D gradient with rotation applied to the gradient vector itself
__device__ __forceinline__ float3 gradient3(int x, int y, int z, int seed, const float *rotation = nullptr) {
    // Generate base gradient (same as before)
    float h1 = hash_int(x, y, z, seed) / 1073741824.0f;
    float h2 = hash_int(z, x, y, seed + 1337) / 1073741824.0f;

    float theta = h1 * 2.0f * 3.14159265f;
    float phi = h2 * 3.14159265f;

    float sin_phi = sinf(phi);
    float3 base_gradient = make_float3(
        cosf(theta) * sin_phi,
        sinf(theta) * sin_phi,
        cosf(phi));

    // Apply rotation to the gradient vector
    if (rotation) {
        base_gradient = rotate(base_gradient, rotation);
    }

    return base_gradient;
}

// Update gradient_noise3 to pass rotation to gradient3
__device__ float gradient_noise3(float x, float y, float z,
                                 int period_x, int period_y, int period_z,
                                 int seed,
                                 const float *rotation = nullptr) {
    int xi = (int)floorf(x);
    int yi = (int)floorf(y);
    int zi = (int)floorf(z);

    float xf = x - xi;
    float yf = y - yi;
    float zf = z - zi;

    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);

    int xi0 = posmod(xi, period_x);
    int yi0 = posmod(yi, period_y);
    int zi0 = posmod(zi, period_z);
    int xi1 = posmod(xi + 1, period_x);
    int yi1 = posmod(yi + 1, period_y);
    int zi1 = posmod(zi + 1, period_z);

    // Get gradients at cube corners WITH rotation applied
    float3 g000 = gradient3(xi0, yi0, zi0, seed, rotation);
    float3 g100 = gradient3(xi1, yi0, zi0, seed, rotation);
    float3 g010 = gradient3(xi0, yi1, zi0, seed, rotation);
    float3 g110 = gradient3(xi1, yi1, zi0, seed, rotation);
    float3 g001 = gradient3(xi0, yi0, zi1, seed, rotation);
    float3 g101 = gradient3(xi1, yi0, zi1, seed, rotation);
    float3 g011 = gradient3(xi0, yi1, zi1, seed, rotation);
    float3 g111 = gradient3(xi1, yi1, zi1, seed, rotation);

    // Distance vectors
    float3 d000 = make_float3(xf, yf, zf);
    float3 d100 = make_float3(xf - 1.0f, yf, zf);
    float3 d010 = make_float3(xf, yf - 1.0f, zf);
    float3 d110 = make_float3(xf - 1.0f, yf - 1.0f, zf);
    float3 d001 = make_float3(xf, yf, zf - 1.0f);
    float3 d101 = make_float3(xf - 1.0f, yf, zf - 1.0f);
    float3 d011 = make_float3(xf, yf - 1.0f, zf - 1.0f);
    float3 d111 = make_float3(xf - 1.0f, yf - 1.0f, zf - 1.0f);

    // Dot products
    float v000 = dot(g000, d000);
    float v100 = dot(g100, d100);
    float v010 = dot(g010, d010);
    float v110 = dot(g110, d110);
    float v001 = dot(g001, d001);
    float v101 = dot(g101, d101);
    float v011 = dot(g011, d011);
    float v111 = dot(g111, d111);

    // Interpolate
    float x00 = lerp(v000, v100, u);
    float x10 = lerp(v010, v110, u);
    float x01 = lerp(v001, v101, u);
    float x11 = lerp(v011, v111, u);

    float y0 = lerp(x00, x10, v);
    float y1 = lerp(x01, x11, v);

    return lerp(y0, y1, w);
}

// Create rotation matrix from Euler angles (in radians)
// pitch = rotation around X-axis
// yaw   = rotation around Y-axis
// roll  = rotation around Z-axis
// Returns row-major 3x3 matrix as Float9
Float9 make_rotation(float pitch, float yaw, float roll) {
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cr = std::cos(roll), sr = std::sin(roll);

    // ZYX rotation order (roll * yaw * pitch)
    return Float9{{cy * cr,
                   cy * sr * sp - sy * cp,
                   cy * sr * cp + sy * sp,
                   sy * cr,
                   sy * sr * sp + cy * cp,
                   sy * sr * cp - cy * sp,
                   -sr,
                   cr * sp,
                   cr * cp}};
}

// Simple Z-axis only rotation (for 2D in-plane rotation)
Float9 make_rotation_z(float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);

    return Float9{{c, -s, 0.0f,
                   s, c, 0.0f,
                   0.0f, 0.0f, 1.0f}};
}

__global__ void generate_noise(
    const Parameters *pars,
    const int width, const int height,
    float *out) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const int idx = y * width + x;

    // Key fix: scale determines the noise frequency
    // The period should match the scaled coordinate space
    float fx = x * pars->_scale; // fx should now be x in "noise space"
    float fy = y * pars->_scale;

    fx += pars->x * pars->period; // in theory this will make 0.5 shift the noise by half of image
    fy += pars->y * pars->period;

    switch (pars->type) {
    case 1:
        out[idx] = gradient_noise3(
            fx, fy, pars->z,
            pars->period, pars->period, pars->period,
            pars->seed);
        break;

    case 0:

        out[idx] = gradient_noise3(
            fx, fy, pars->z,
            pars->period, pars->period, pars->period,
            pars->seed, pars->_rotation_matrix);

        break;
    }
}

void TEMPLATE_CLASS_NAME::allocate_device() {
}

void TEMPLATE_CLASS_NAME::deallocate_device() {

    // DeviceArray2D's
#ifdef TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#define X(TYPE, NAME, DESCRIPTION) \
    NAME.free_device();
    TEMPLATE_CLASS_DEVICE_ARRAY_2DS
#undef X
#endif
}

void TEMPLATE_CLASS_NAME::process() {

    noise.set_stream(stream.get());
    noise.resize(pars.width, pars.height);

    Float9 rot = make_rotation(pars.rotate_x, pars.rotate_y, pars.rotate_z);
    // Float9 rot = make_rotation_z(pars.rotate_z);
    std::copy(rot.begin(), rot.end(), pars._rotation_matrix);

    pars._scale = static_cast<float>(pars.period) / pars.width;

    
    refresh_device_config();
    generate_noise<<<grid, block, 0, stream.get()>>>(dev_pars.dev_ptr(), pars.width, pars.height, noise.dev_ptr());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

} // namespace TEMPLATE_NAMESPACE

#include "template_macro_undef.h"
