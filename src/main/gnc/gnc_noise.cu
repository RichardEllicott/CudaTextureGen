#include "core/cuda/math.cuh"
#include "gnc/gnc_noise.cuh"

namespace TEMPLATE_NAMESPACE {

using namespace core::cuda::math;

// quintic smoothstep
// aka Perlin’s fade function
// Creates an S-curve (sigmoid-like shape)
__device__ __forceinline__ float quintic_smoothstep(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

// 🧪  this could be cheaper than quintic_smoothstep, but less perfect
__device__ __forceinline__ float smoothstep3(float t) { return t * t * (3 - 2 * t); }

// 3D gradient for gradient noise (looks like simplex)
__device__ __forceinline__ float3 gradient3(int x, int y, int z, int seed) {
    // Hash to get a pseudo-random angle and elevation
    float h1 = hash_int(x, y, z, seed) / 1073741824.0f; // range ~[0, 2]
    float h2 = hash_int(z, x, y, seed + 1337) / 1073741824.0f;

    // Convert to spherical coordinates
    float theta = h1 * 2.0f * 3.14159265f; // azimuthal angle
    float phi = h2 * 3.14159265f;          // polar angle

    float sin_phi = sinf(phi);
    return {cosf(theta) * sin_phi, sinf(theta) * sin_phi, cosf(phi)};
}

// 3D Gradient Noise
__device__ float gradient_noise3(float x, float y, float z,
                                 int period_x, int period_y, int period_z,
                                 int seed) {
    int xi = (int)floorf(x);
    int yi = (int)floorf(y);
    int zi = (int)floorf(z);

    float xf = x - xi;
    float yf = y - yi;
    float zf = z - zi;

    float u = quintic_smoothstep(xf);
    float v = quintic_smoothstep(yf);
    float w = quintic_smoothstep(zf);

    int xi0 = posmod(xi, period_x);
    int yi0 = posmod(yi, period_y);
    int zi0 = posmod(zi, period_z);
    int xi1 = posmod(xi + 1, period_x);
    int yi1 = posmod(yi + 1, period_y);
    int zi1 = posmod(zi + 1, period_z);

    // Get gradients at cube corners
    float3 g000 = gradient3(xi0, yi0, zi0, seed);
    float3 g100 = gradient3(xi1, yi0, zi0, seed);
    float3 g010 = gradient3(xi0, yi1, zi0, seed);
    float3 g110 = gradient3(xi1, yi1, zi0, seed);
    float3 g001 = gradient3(xi0, yi0, zi1, seed);
    float3 g101 = gradient3(xi1, yi0, zi1, seed);
    float3 g011 = gradient3(xi0, yi1, zi1, seed);
    float3 g111 = gradient3(xi1, yi1, zi1, seed);

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

__global__ void generate_gradient_noise3(
    const int width, const int height,
    float *out,
    float3 scale,
    float3 period,
    float3 offset,
    int seed) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const int idx = y * width + x;

    // Key fix: scale determines the noise frequency
    // The period should match the scaled coordinate space
    float fx = x * scale.x; // fx should now be x in "noise space"
    float fy = y * scale.y;

    fx += offset.x * period.x; // in theory this will make 0.5 shift the noise by half of image
    fy += offset.y * period.y;

    out[idx] = gradient_noise3(
        fx, fy, offset.z,
        period.x, period.y, period.z,
        seed);
}

void TEMPLATE_CLASS_NAME::process() {

    output.instantiate_if_null(); // ensure output
    stream.instantiate_if_null(); // ensure stream
    output->resize(width, height);

    float3 scale = {period[0] / width, period[1] / height, 1.0f};

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    generate_gradient_noise3<<<grid, block, 0, stream->get()>>>(
        width, height,
        output->dev_ptr(),
        scale,
        to_float3(period),
        to_float3(offset),
        seed);
}

} // namespace TEMPLATE_NAMESPACE
