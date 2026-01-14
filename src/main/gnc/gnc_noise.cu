#include "core/cuda/cast.cuh"
#include "core/cuda/math.cuh"
#include "gnc/gnc_noise.cuh"

namespace TEMPLATE_NAMESPACE {

// using namespace core::cuda::math;

// // 3D gradient for gradient noise (looks like simplex)
// __device__ __forceinline__ float3 gradient3(const int x, const int y, const int z, const int seed) {
//     // Hash to get a pseudo-random angle and elevation
//     // float h1 = hash_int(x, y, z, seed) / 1073741824.0f; // range ~[0, 2]
//     // float h2 = hash_int(z, x, y, seed + 1337) / 1073741824.0f;

//     // ⚠️ the maths here is mistaken i think!?!

//     constexpr float INV_U30 = 0x1p-30f; // exactly 2^-30
//     float h1 = hash_int(x, y, z, seed) * INV_U30;
//     float h2 = hash_int(z, x, y, seed + 1337) * INV_U30;

//     // Convert to spherical coordinates
//     float theta = h1 * 2.0f * 3.14159265f; // azimuthal angle
//     float phi = h2 * 3.14159265f;          // polar angle

//     float sin_phi = sinf(phi);
//     return {cosf(theta) * sin_phi, sinf(theta) * sin_phi, cosf(phi)}; // could use fast math
// }

namespace cmath = core::cuda::math;

// ⚠️ update to faster maths and more correct

// Generates a deterministic pseudo‑random unit vector in 3D using hashed spherical coordinates.
D_INLINE float3 gradient3(int x, int y, int z, int seed) {

    //  ⚠️ could concider using one hash and my mixing function
    float u = cmath::hash_float(x, y, z, seed);               // ∈ [0,1) or (0 <= u < 1)
    float v = cmath::hash_float_signed(z, x, y, seed + 1337); // ∈ [-1,1)

    // Spherical coordinates:
    float theta = u * cmath::TAU; // theta ∈ [0, 2π)
    float phi = acosf(v);         // phi   ∈ [0, π]

    // Fast sincos for both angles
    float sin_theta, cos_theta;
    cmath::fast_sincosf(theta, &sin_theta, &cos_theta);

    float sin_phi, cos_phi;
    cmath::fast_sincosf(phi, &sin_phi, &cos_phi);

    // Convert spherical → Cartesian
    return {cos_theta * sin_phi,
            sin_theta * sin_phi,
            cos_phi};
}

// 3D Gradient Noise
D_INLINE float gradient_noise3(
    const float3 position,

    int3 period,

    int3 wrap,

    const int seed

) {
    // ================================================================
    int xi = (int)floorf(position.x);
    int yi = (int)floorf(position.y);
    int zi = (int)floorf(position.z);

    // fractions
    float xf = position.x - xi;
    float yf = position.y - yi;
    float zf = position.z - zi;

    // smoothing
    float u = cmath::quintic_smoothstep(xf);
    float v = cmath::quintic_smoothstep(yf);
    float w = cmath::quintic_smoothstep(zf);

    // wrap or not
    // bool wrap_x = true;
    // bool wrap_y = true;
    // bool wrap_z = true;

    int xi0 = wrap.x ? cmath::posmod(xi, period.x) : xi;
    int xi1 = wrap.x ? cmath::posmod(xi + 1, period.x) : (xi + 1);

    int yi0 = wrap.y ? cmath::posmod(yi, period.y) : yi;
    int yi1 = wrap.y ? cmath::posmod(yi + 1, period.y) : (yi + 1);

    int zi0 = wrap.z ? cmath::posmod(zi, period.z) : zi;
    int zi1 = wrap.z ? cmath::posmod(zi + 1, period.z) : (zi + 1);

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
    float v000 = cmath::dot(g000, d000);
    float v100 = cmath::dot(g100, d100);
    float v010 = cmath::dot(g010, d010);
    float v110 = cmath::dot(g110, d110);
    float v001 = cmath::dot(g001, d001);
    float v101 = cmath::dot(g101, d101);
    float v011 = cmath::dot(g011, d011);
    float v111 = cmath::dot(g111, d111);

    // Interpolate
    float x00 = cmath::lerp(v000, v100, u);
    float x10 = cmath::lerp(v010, v110, u);
    float x01 = cmath::lerp(v001, v101, u);
    float x11 = cmath::lerp(v011, v111, u);

    float y0 = cmath::lerp(x00, x10, v);
    float y1 = cmath::lerp(x01, x11, v);

    return cmath::lerp(y0, y1, w);
}

//
//
//

//
//
//

__global__ void generate_gradient_noise3(
    const int2 size,
    float *__restrict__ out,

    const float3 scale,
    const float3 period,
    const float3 offset,
    const int3 wrap,

    const int seed) {
    // ================================================================
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size.x || y >= size.y) return;
    const int idx = y * size.x + x;
    // ================================================================

    // Key fix: scale determines the noise frequency
    // The period should match the scaled coordinate space
    float fx = x * scale.x; // fx should now be x in "noise space"
    float fy = y * scale.y;

    fx += offset.x * period.x; // in theory this will make 0.5 shift the noise by half of image
    fy += offset.y * period.y;

    float3 position = {fx, fy, offset.z};

    // cast::to_intN

    out[idx] = gradient_noise3(
        position,
        core::cuda::cast::to_int3(period),
        wrap,
        seed);
}

void TEMPLATE_CLASS_NAME::_compute() {

    output.instantiate_if_null(); // ensure output
    stream.instantiate_if_null(); // ensure stream
    output->resize(size.x, size.y);

    float3 scale = {period.x / size.x, period.y / size.y, 1.0f};

    dim3 block(16, 16);
    dim3 grid((size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y);

    generate_gradient_noise3<<<grid, block, 0, stream->get()>>>(
        size,
        output->dev_ptr(),
        scale,
        period,
        offset,
        core::cuda::cast::to_int3(wrap),
        seed);
}

} // namespace TEMPLATE_NAMESPACE
