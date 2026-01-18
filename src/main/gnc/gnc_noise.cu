// #include "core/cuda/math.cuh"
#include "core/cuda/math/transform.cuh"
#include "gnc/gnc_noise.cuh"

namespace TEMPLATE_NAMESPACE {

namespace cmath = core::cuda::math;

#pragma region HELPERS

// Generates a deterministic pseudo‑random unit vector in 3D using hashed spherical coordinates.
DH_INLINE float3 gradient3(int x, int y, int z, int seed) {

    // generate two random floats with hash then a hash mix
    uint32_t hash = cmath::hash_uint(x, y, z, seed);
    float u = cmath::hash_float(hash); //  ∈ [0,1)]
    hash = cmath::hash_mix(hash);
    float v = cmath::hash_float_signed(hash); // ∈ [-1,1)]

    // Spherical coordinates:
    float theta = u * cmath::TAU; // theta ∈ [0, 2π)
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
    int3 period,
    int3 wrap, // 0 = false, 1 = true
    const int seed,
    int smoothing_mode // 0 none, 1 cubic, 2 quintic

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
    float3 g000 = gradient3(xi0, yi0, zi0, seed);
    float3 g100 = gradient3(xi1, yi0, zi0, seed);
    float3 g010 = gradient3(xi0, yi1, zi0, seed);
    float3 g110 = gradient3(xi1, yi1, zi0, seed);
    float3 g001 = gradient3(xi0, yi0, zi1, seed);
    float3 g101 = gradient3(xi1, yi0, zi1, seed);
    float3 g011 = gradient3(xi0, yi1, zi1, seed);
    float3 g111 = gradient3(xi1, yi1, zi1, seed);

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
    int smoothing_mode,
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
    float3 g000 = basis * gradient3(xi0, yi0, zi0, seed);
    float3 g100 = basis * gradient3(xi1, yi0, zi0, seed);
    float3 g010 = basis * gradient3(xi0, yi1, zi0, seed);
    float3 g110 = basis * gradient3(xi1, yi1, zi0, seed);
    float3 g001 = basis * gradient3(xi0, yi0, zi1, seed);
    float3 g101 = basis * gradient3(xi1, yi0, zi1, seed);
    float3 g011 = basis * gradient3(xi0, yi1, zi1, seed);
    float3 g111 = basis * gradient3(xi1, yi1, zi1, seed);

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

#pragma endregion

#pragma region KERNELS

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

__global__ void generate_gradient_noise3(
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

    // Key fix: scale determines the noise frequency
    // The period should match the scaled coordinate space
    float fx = pos.x * scale.x; // fx should now be x in "noise space"
    float fy = pos.y * scale.y;

    fx += offset.x * period.x; // in theory this will make 0.5 shift the noise by half of image
    fy += offset.y * period.y;

    float3 position = {fx, fy, offset.z};

    out[idx] = gradient_noise3(
        position,
        to_int3(period),
        wrap,
        seed,
        smoothing_mode,
        basis);
}

#pragma endregion

#pragma region MAIN

void TEMPLATE_CLASS_NAME::_compute() {

    output.instantiate_if_null(); // ensure output
    stream.instantiate_if_null(); // ensure stream
    output->resize(size.x, size.y);

    float3 scale = {period.x / size.x, period.y / size.y, 1.0f};

    dim3 block(16, 16);
    dim3 grid((size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y);

    printf("rotation = (%f, %f, %f)\n", rotation.x, rotation.y, rotation.z);
    printf("⚠️ rotation still in experimental!\n");

    auto basis = cmath::Basis3(rotation);

    generate_gradient_noise3<<<grid, block, 0, stream->get()>>>(
        size,
        output->dev_ptr(),
        scale,
        period,
        offset,
        to_int3(wrap),
        seed,
        smoothing_mode,
        basis);

    // generate_gradient_noise3<<<grid, block, 0, stream->get()>>>(
    //     size,
    //     output->dev_ptr(),
    //     scale,
    //     period,
    //     offset,
    //     to_int3(wrap),
    //     seed,
    //     smoothing_mode,
    //     cmath::basis3(rotation));
}

#pragma endregion

} // namespace TEMPLATE_NAMESPACE
