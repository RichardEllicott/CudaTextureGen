#include "noise.cuh"
#include "noise_util.cuh"

using namespace noise_util;

namespace TEMPLATE_NAMESPACE {

#pragma region VALUE_NOISE

// 2D Value Noise
__device__ float value_noise2(float x, float y, int period_x, int period_y, int seed) {
    int xi = (int)floorf(x);
    int yi = (int)floorf(y);

    float xf = x - floorf(x);
    float yf = y - floorf(y);

    float u = fade(xf);
    float v = fade(yf);

    // Wrap grid coordinates to the period
    int xi0 = posmod(xi, period_x);
    int yi0 = posmod(yi, period_y);
    int xi1 = posmod(xi + 1, period_x);
    int yi1 = posmod(yi + 1, period_y);

    float a = hash_scalar(xi0, yi0, seed);
    float b = hash_scalar(xi1, yi0, seed);
    float c = hash_scalar(xi0, yi1, seed);
    float d = hash_scalar(xi1, yi1, seed);

    float x1 = lerp(a, b, u);
    float x2 = lerp(c, d, u);

    return lerp(x1, x2, v);
}

#pragma endregion

#pragma region GRADIENT_NOISE

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

    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);

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

#pragma endregion

__global__ void generate_noise(
    float *out,
    const size_t width, const size_t height,
    const Parameters *const __restrict__ pars) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const int idx = y * width + x;

    // Pixels -> noise coords where 'period' is the lattice period across the image.
    float fx = (float)x * ((float)pars->period_x / (float)pars->width);
    float fy = (float)y * ((float)pars->period_y / (float)pars->height);

    float fz = pars->z * pars->period_z;

    // Scroll offsets in lattice units
    fx += pars->x * pars->period_x;
    fy += pars->y * pars->period_y;

    switch (pars->type) {
    case 0:
        out[idx] = gradient_noise3(
            fx, fy, fz,
            pars->period_x, pars->period_y, pars->period_z,
            pars->seed);
        break;
    case 1:
        out[idx] = value_noise2(
            fx, fy,
            pars->period_x, pars->period_y,
            pars->seed);
        break;
    }
}

void TEMPLATE_CLASS_NAME::process() {

    // pars.scale = static_cast<float>(pars.period) / pars.width;

    image.resize(pars.width, pars.height);
    image.allocate_device(); // just allocate

    core::cuda::Stream stream;

    core::cuda::Struct<Parameters> _pars(pars); // automaticly uploads and free

    dim3 block(pars._block, pars._block);
    dim3 grid((pars.width + block.x - 1) / block.x,
              (pars.height + block.y - 1) / block.y);

    generate_noise<<<grid, block, 0, stream.get()>>>(
        image.dev_ptr(),
        pars.width, pars.height,
        _pars.dev_ptr());

    stream.sync();

    image.download();
    image.free_device();
}

} // namespace TEMPLATE_NAMESPACE
