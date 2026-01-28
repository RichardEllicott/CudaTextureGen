// #include "core/cuda/math.cuh"
#include "core/cuda/math/transform.cuh"
#include "gnc/gnc_noise.cuh"

namespace TEMPLATE_NAMESPACE {

namespace cmath = core::cuda::math;

#pragma region VALUE_NOISE

// 2D Value Noise
__device__ float value_noise2(float x, float y, int period_x, int period_y, int seed) {
    int xi = (int)floorf(x);
    int yi = (int)floorf(y);

    float xf = x - floorf(x);
    float yf = y - floorf(y);

    float u = cmath::smooth::cubic(xf);
    float v = cmath::smooth::cubic(yf);

    // Wrap grid coordinates to the period
    int xi0 = cmath::posmod(xi, period_x);
    int yi0 = cmath::posmod(yi, period_y);
    int xi1 = cmath::posmod(xi + 1, period_x);
    int yi1 = cmath::posmod(yi + 1, period_y);

    float a = chash::hash_float_signed(xi0, yi0, seed);
    float b = chash::hash_float_signed(xi1, yi0, seed);
    float c = chash::hash_float_signed(xi0, yi1, seed);
    float d = chash::hash_float_signed(xi1, yi1, seed);

    float x1 = cmath::lerp(a, b, u);
    float x2 = cmath::lerp(c, d, u);

    return cmath::lerp(x1, x2, v);
}

// 2D Value Noise (with rotation) .... BROKEN!!
__device__ float value_noise2(float x, float y, int period_x, int period_y, int seed, float angle) {
    // Compute center of the noise domain
    float cx = 0.5f * period_x;
    float cy = 0.5f * period_y;

    // Shift coordinates so rotation is centered
    float x_shifted = x - cx;
    float y_shifted = y - cy;

    // Apply rotation
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);
    float x_rot = cos_a * x_shifted - sin_a * y_shifted + cx;
    float y_rot = sin_a * x_shifted + cos_a * y_shifted + cy;

    // Compute grid cell and fractional offset from rotated coordinates
    int xi = (int)floorf(x_rot);
    int yi = (int)floorf(y_rot);

    float xf = x_rot - floorf(x_rot);
    float yf = y_rot - floorf(y_rot);

    // Interpolation weights
    float u = cmath::smooth::cubic(xf);
    float v = cmath::smooth::cubic(yf);

    // Wrap grid coordinates to the period
    int xi0 = cmath::posmod(xi, period_x);
    int yi0 = cmath::posmod(yi, period_y);
    int xi1 = cmath::posmod(xi + 1, period_x);
    int yi1 = cmath::posmod(yi + 1, period_y);

    // Sample scalar values at corners
    float a = chash::hash_float_signed(xi0, yi0, seed);
    float b = chash::hash_float_signed(xi1, yi0, seed);
    float c = chash::hash_float_signed(xi0, yi1, seed);
    float d = chash::hash_float_signed(xi1, yi1, seed);

    // Interpolate
    float x1 = cmath::lerp(a, b, u);
    float x2 = cmath::lerp(c, d, u);
    return cmath::lerp(x1, x2, v);
}

#pragma endregion

#pragma region NOISE2

// DH_INLINE float2 noise_gradient2(int x, int y, int seed, float angle) {
//     float theta = chash::hash_float(x, y, 0, seed) * cmath::TAU + angle;
//     float s, c;
//     cmath::fast::sincosf(theta, &s, &c);
//     return make_float2(c, s);
// }

// old one
DH_INLINE float2 noise_gradient2(int x, int y, int seed, float angle) {
    float raw_angle = (chash::hash_int(x, y, 0, seed) / 1073741824.0f) * 2.0f * 3.14159265f;
    float final_angle = raw_angle + angle;

    return make_float2(cosf(final_angle), sinf(final_angle));
}

// 2D Gradient Noise
DH_INLINE float gradient_noise2(
    const float2 pos,
    const int2 period,
    const int2 wrap,
    const int seed,
    const float angle) {
    // ================================================================

    int xi = cmath::floor(pos.x);
    int yi = cmath::floor(pos.y);

    float xf = pos.x - xi;
    float yf = pos.y - yi;

    float u = cmath::smooth::cubic(xf);
    float v = cmath::smooth::cubic(yf);

    // int xi0 = cmath::posmod(xi, period.x);
    // int yi0 = cmath::posmod(yi, period.y);
    // int xi1 = cmath::posmod(xi + 1, period.x);
    // int yi1 = cmath::posmod(yi + 1, period.y);

    // with optional wrap
    int xi0 = wrap.x ? cmath::posmod(xi, period.x) : xi;
    int xi1 = wrap.x ? cmath::posmod(xi + 1, period.x) : (xi + 1);
    int yi0 = wrap.y ? cmath::posmod(yi, period.y) : yi;
    int yi1 = wrap.y ? cmath::posmod(yi + 1, period.y) : (yi + 1);

    // Get gradients at corners
    float2 g00 = noise_gradient2(xi0, yi0, seed, angle);
    float2 g10 = noise_gradient2(xi1, yi0, seed, angle);
    float2 g01 = noise_gradient2(xi0, yi1, seed, angle);
    float2 g11 = noise_gradient2(xi1, yi1, seed, angle);

    // Calculate dot products with distance vectors
    float d00 = g00.x * xf + g00.y * yf;
    float d10 = g10.x * (xf - 1.0f) + g10.y * yf;
    float d01 = g01.x * xf + g01.y * (yf - 1.0f);
    float d11 = g11.x * (xf - 1.0f) + g11.y * (yf - 1.0f);

    // Interpolate
    float x1 = cmath::lerp(d00, d10, u);
    float x2 = cmath::lerp(d01, d11, u);

    float noise_value = cmath::lerp(x1, x2, v); // likely will not exceed 2**-0.5 (0.7071067811865476)
    noise_value *= 1.4f;                        // should bring the noise closer to -1 to 1 range

    return noise_value;
}

__global__ void gradient_noise2_kernel(
    const int2 size,
    float *__restrict__ out,

    const float2 scale,
    const float2 period,
    const float2 offset,
    const int2 wrap,

    const int seed,
    const float angle) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= size.x || pos.y >= size.y) return;
    int idx = cmath::pos_to_idx(pos, size);
    // ================================================================

    float2 noise_pos = make_float2(
        pos.x * scale.x,
        pos.y * scale.y);

    // offset of 0.5 will be scaled for half the image
    noise_pos += make_float2(
        offset.x * period.x,
        offset.y * period.y);

    out[idx] = gradient_noise2(
        noise_pos,
        to_int2(period),
        wrap,
        seed,
        angle);
}

#pragma endregion

#pragma region NOISE3

// Generates a deterministic pseudo‑random unit vector in 3D using hashed spherical coordinates.
DH_INLINE float3 noise_gradient3(int x, int y, int z, int seed) {

    // generate two random floats with hash then a hash mix
    uint32_t hash = chash::hash_uint(x, y, z, seed);
    float u = chash::hash_float(hash); //  ∈ [0,1)]
    hash = chash::hash_mix(hash);
    float v = chash::hash_float_signed(hash); // ∈ [-1,1)]

    // Spherical coordinates:
    float theta = u * cmath::constants::TAU; // theta ∈ [0, 2π)
    float phi = acosf(v);         // phi   ∈ [0, π]

    // Fast sincos for both angles
    float sin_theta, cos_theta;
    cmath::fast::sincosf(theta, &sin_theta, &cos_theta);

    float sin_phi, cos_phi;
    cmath::fast::sincosf(phi, &sin_phi, &cos_phi);

    // Convert spherical → Cartesian
    return make_float3(
        cos_theta * sin_phi,
        sin_theta * sin_phi,
        cos_phi);
}

// 3D Gradient Noise
DH_INLINE float gradient_noise3(
    const float3 position,
    const int3 period,
    const int3 wrap, // 0 = false, 1 = true
    const int seed,
    const int smoothing_mode // 0 none, 1 cubic, 2 quintic

) {
    // ================================================================
    int3 grid = to_int3(cmath::floor(position));
    float3 fract = position - to_float3(grid);

    int3 smoothing_mode3 = {smoothing_mode, smoothing_mode, smoothing_mode};

    // with optional wrap
    int xi0 = wrap.x ? cmath::posmod(grid.x, period.x) : grid.x;
    int xi1 = wrap.x ? cmath::posmod(grid.x + 1, period.x) : (grid.x + 1);
    int yi0 = wrap.y ? cmath::posmod(grid.y, period.y) : grid.y;
    int yi1 = wrap.y ? cmath::posmod(grid.y + 1, period.y) : (grid.y + 1);
    int zi0 = wrap.z ? cmath::posmod(grid.z, period.z) : grid.z;
    int zi1 = wrap.z ? cmath::posmod(grid.z + 1, period.z) : (grid.z + 1);

    // Get gradients at cube corners
    float3 g000 = noise_gradient3(xi0, yi0, zi0, seed);
    float3 g100 = noise_gradient3(xi1, yi0, zi0, seed);
    float3 g010 = noise_gradient3(xi0, yi1, zi0, seed);
    float3 g110 = noise_gradient3(xi1, yi1, zi0, seed);
    float3 g001 = noise_gradient3(xi0, yi0, zi1, seed);
    float3 g101 = noise_gradient3(xi1, yi0, zi1, seed);
    float3 g011 = noise_gradient3(xi0, yi1, zi1, seed);
    float3 g111 = noise_gradient3(xi1, yi1, zi1, seed);

    // Distance vectors
    float3 d000 = make_float3(fract.x, fract.y, fract.z);
    float3 d100 = make_float3(fract.x - 1.0f, fract.y, fract.z);
    float3 d010 = make_float3(fract.x, fract.y - 1.0f, fract.z);
    float3 d110 = make_float3(fract.x - 1.0f, fract.y - 1.0f, fract.z);
    float3 d001 = make_float3(fract.x, fract.y, fract.z - 1.0f);
    float3 d101 = make_float3(fract.x - 1.0f, fract.y, fract.z - 1.0f);
    float3 d011 = make_float3(fract.x, fract.y - 1.0f, fract.z - 1.0f);
    float3 d111 = make_float3(fract.x - 1.0f, fract.y - 1.0f, fract.z - 1.0f);

    // Dot products
    float v000 = cmath::dot(g000, d000);
    float v100 = cmath::dot(g100, d100);
    float v010 = cmath::dot(g010, d010);
    float v110 = cmath::dot(g110, d110);
    float v001 = cmath::dot(g001, d001);
    float v101 = cmath::dot(g101, d101);
    float v011 = cmath::dot(g011, d011);
    float v111 = cmath::dot(g111, d111);

    // // smoothing
    // float3 uvw = make_float3(
    //     cmath::smooth::apply_smoothing(fract.x, smoothing_mode3.x),
    //     cmath::smooth::apply_smoothing(fract.y, smoothing_mode3.y),
    //     cmath::smooth::apply_smoothing(fract.z, smoothing_mode3.z));

    fract = make_float3(
        cmath::smooth::apply_smoothing(fract.x, smoothing_mode3.x),
        cmath::smooth::apply_smoothing(fract.y, smoothing_mode3.y),
        cmath::smooth::apply_smoothing(fract.z, smoothing_mode3.z));

    // Interpolate
    float x00 = cmath::lerp(v000, v100, fract.x);
    float x10 = cmath::lerp(v010, v110, fract.x);
    float x01 = cmath::lerp(v001, v101, fract.x);
    float x11 = cmath::lerp(v011, v111, fract.x);

    float y0 = cmath::lerp(x00, x10, fract.y);
    float y1 = cmath::lerp(x01, x11, fract.y);

    return cmath::lerp(y0, y1, fract.z);
}

DH_INLINE float gradient_noise3(
    const float3 position,
    const int3 period,
    const int3 wrap, // 0 = false, 1 = true
    const int seed,
    const int smoothing_mode,
    const cmath::Basis3 &basis // NEW: rotation basis (identity by default)
) {
    // ================================================================
    // Grid + fractional position (domain NOT rotated)
    int3 grid = to_int3(cmath::floor(position));
    float3 fract = position - to_float3(grid);

    int3 smoothing_mode3 = {smoothing_mode, smoothing_mode, smoothing_mode};

    // Wrapped lattice coordinates
    int xi0 = wrap.x ? cmath::posmod(grid.x, period.x) : grid.x;
    int xi1 = wrap.x ? cmath::posmod(grid.x + 1, period.x) : (grid.x + 1);
    int yi0 = wrap.y ? cmath::posmod(grid.y, period.y) : grid.y;
    int yi1 = wrap.y ? cmath::posmod(grid.y + 1, period.y) : (grid.y + 1);
    int zi0 = wrap.z ? cmath::posmod(grid.z, period.z) : grid.z;
    int zi1 = wrap.z ? cmath::posmod(grid.z + 1, period.z) : (grid.z + 1);

    // ================================================================
    // Gradients at cube corners (ROTATED by basis)
    float3 g000 = basis * noise_gradient3(xi0, yi0, zi0, seed);
    float3 g100 = basis * noise_gradient3(xi1, yi0, zi0, seed);
    float3 g010 = basis * noise_gradient3(xi0, yi1, zi0, seed);
    float3 g110 = basis * noise_gradient3(xi1, yi1, zi0, seed);
    float3 g001 = basis * noise_gradient3(xi0, yi0, zi1, seed);
    float3 g101 = basis * noise_gradient3(xi1, yi0, zi1, seed);
    float3 g011 = basis * noise_gradient3(xi0, yi1, zi1, seed);
    float3 g111 = basis * noise_gradient3(xi1, yi1, zi1, seed);

    // ================================================================
    // Distance vectors (unchanged)
    float3 d000 = make_float3(fract.x, fract.y, fract.z);
    float3 d100 = make_float3(fract.x - 1.0f, fract.y, fract.z);
    float3 d010 = make_float3(fract.x, fract.y - 1.0f, fract.z);
    float3 d110 = make_float3(fract.x - 1.0f, fract.y - 1.0f, fract.z);
    float3 d001 = make_float3(fract.x, fract.y, fract.z - 1.0f);
    float3 d101 = make_float3(fract.x - 1.0f, fract.y, fract.z - 1.0f);
    float3 d011 = make_float3(fract.x, fract.y - 1.0f, fract.z - 1.0f);
    float3 d111 = make_float3(fract.x - 1.0f, fract.y - 1.0f, fract.z - 1.0f);

    // ================================================================
    // Dot products
    float v000 = cmath::dot(g000, d000);
    float v100 = cmath::dot(g100, d100);
    float v010 = cmath::dot(g010, d010);
    float v110 = cmath::dot(g110, d110);
    float v001 = cmath::dot(g001, d001);
    float v101 = cmath::dot(g101, d101);
    float v011 = cmath::dot(g011, d011);
    float v111 = cmath::dot(g111, d111);

    // ================================================================
    // Smoothing
    fract = make_float3(
        cmath::smooth::apply_smoothing(fract.x, smoothing_mode3.x),
        cmath::smooth::apply_smoothing(fract.y, smoothing_mode3.y),
        cmath::smooth::apply_smoothing(fract.z, smoothing_mode3.z));

    // ================================================================
    // Interpolation
    float x00 = cmath::lerp(v000, v100, fract.x);
    float x10 = cmath::lerp(v010, v110, fract.x);
    float x01 = cmath::lerp(v001, v101, fract.x);
    float x11 = cmath::lerp(v011, v111, fract.x);

    float y0 = cmath::lerp(x00, x10, fract.y);
    float y1 = cmath::lerp(x01, x11, fract.y);

    return cmath::lerp(y0, y1, fract.z);
}

__global__ void gradient_noise3_kernel(
    const int2 size,
    float *__restrict__ out,

    const float3 scale,
    const float3 period,
    const float3 offset,
    const int3 wrap,

    const int seed,
    const int smoothing_mode,
    const cmath::Basis3 basis = cmath::Basis3()) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= size.x || pos.y >= size.y) return;
    int idx = cmath::pos_to_idx(pos, size);
    // ================================================================

    // // Key fix: scale determines the noise frequency
    // // The period should match the scaled coordinate space
    // float fx = pos.x * scale.x; // fx should now be x in "noise space"
    // float fy = pos.y * scale.y;

    // scale determines the noise frequency, it is calculated before we launch the kernel
    float3 noise_pos = make_float3(
        pos.x * scale.x,
        pos.y * scale.y,
        0.0f);

    // offset of 0.5 will be scaled for half the image
    noise_pos += make_float3(
        offset.x * period.x,
        offset.y * period.y,
        offset.z * period.z);

    // fx += offset.x * period.x; // in theory this will make 0.5 shift the noise by half of image
    // fy += offset.y * period.y;

    // float3 position = make_float3(fx, fy, offset.z);

    // float3 position = make_float3(
    //     offset.x * period.x,
    //     offset.y * period.y,
    //     offset.z * period.z);

    out[idx] = gradient_noise3(
        noise_pos,
        to_int3(period),
        wrap,
        seed,
        smoothing_mode,
        basis);
}

#pragma endregion

__global__ void generate_fbm_noise3(
    const int2 size,
    float *__restrict__ out,
    const float3 scale,
    const float3 period,
    const float3 offset,
    const int3 wrap,
    const int seed) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= size.x || pos.y >= size.y) return;
    int idx = cmath::pos_to_idx(pos, size);
    // ================================================================
}

#pragma region MAIN

#pragma region WORLEY

// Returns F1 and F2 distances for 2D Worley noise.
// xi, yi = integer lattice coords of the query point
// xf, yf = fractional coords inside the cell
// period = wrapping period (0 = no wrap)
// seed   = hash seed
DH_INLINE float2 worley2(float2 p, int2 period, int seed) {
    // Integer lattice coordinate
    int xi = floorf(p.x);
    int yi = floorf(p.y);

    // Fractional offset inside the cell
    float2 f = make_float2(p.x - xi, p.y - yi);

    float F1 = 1e9f;
    float F2 = 1e9f;

    // Check 3×3 neighboring cells
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int cx = xi + dx;
            int cy = yi + dy;

            // Optional wrapping
            if (period.x > 0) cx = cmath::posmod(cx, period.x);
            if (period.y > 0) cy = cmath::posmod(cy, period.y);

            // Random point inside this cell
            float2 jitter = chash::hash_float2(cx, cy, 0, seed);
            jitter = jitter * 0.999f; // keep inside cell

            // Offset from query point
            float2 d = make_float2(dx + jitter.x - f.x,
                                   dy + jitter.y - f.y);

            float dist = cmath::dot(d, d); // squared distance

            // Branchless F1/F2 update
            float oldF1 = F1;
            F1 = fminf(F1, dist);
            F2 = fminf(F2, fmaxf(oldF1, dist));
        }
    }

    return make_float2(F1, F2);
}

__global__ void generate_worley(
    int2 size,
    float *out,
    float scale,
    int2 period,
    int seed) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= size.x || pos.y >= size.y) return;
    int idx = cmath::pos_to_idx(pos, size);
    // ================================================================

    // Convert pixel coordinate → noise coordinate
    float2 p = make_float2(pos.x, pos.y) * scale;

    // Compute F1/F2
    float2 f = worley2(p, period, seed);

    // Example: classic Worley F1 noise
    float value = sqrtf(f.x);

    out[idx] = value;
}

#pragma endregion

void TEMPLATE_CLASS_NAME::_compute() {

    output.instantiate_if_null(); // ensure output
    stream.instantiate_if_null(); // ensure stream
    output->resize(size.x, size.y);

    float3 scale = make_float3(
        period.x / size.x,
        period.y / size.y,
        period.z / 1.0f);

    auto basis = cmath::Basis3(rotation);

    dim3 block(16, 16);
    dim3 grid = cmath::calculate_grid(size, block);

    printf("rotation = (%f, %f, %f)\n", rotation.x, rotation.y, rotation.z);
    printf("⚠️ rotation still in experimental!\n");

    switch (mode) {

    case 0: {
        // ================================================================
        gradient_noise3_kernel<<<grid, block, 0, stream->get()>>>(
            size,
            output->dev_ptr(),
            scale,
            period,
            offset,
            to_int3(wrap),
            seed,
            smoothing_mode,
            basis);
        // ----------------------------------------------------------------
    } break;
    case 1: {
        // ================================================================
        gradient_noise2_kernel<<<grid, block, 0, stream->get()>>>(
            size,
            output->dev_ptr(),
            make_float2(scale.x, scale.y),
            make_float2(period.x, period.y),
            make_float2(offset.x, offset.y),
            make_int2(wrap[0], wrap[1]),
            seed,
            rotation.z);
        // ----------------------------------------------------------------
    } break;
    case 2: {
        // ================================================================
        generate_worley<<<grid, block, 0, stream->get()>>>(
            size,
            output->dev_ptr(),
            worley_scale,
            to_int2(make_float2(period.x, period.y)),
            seed);
        // ----------------------------------------------------------------
    } break;
    case 3: {
        // ================================================================
        // ----------------------------------------------------------------
    } break;
    case 4: {
        // ================================================================
        // ----------------------------------------------------------------
    } break;
    default: {
        // ================================================================
        throw std::runtime_error("Invalid noise mode");
        // ----------------------------------------------------------------
    } break;
    }
}

#pragma endregion

} // namespace TEMPLATE_NAMESPACE
