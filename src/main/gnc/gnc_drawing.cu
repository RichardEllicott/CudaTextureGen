#include "gnc/gnc_drawing.cuh"

namespace core::cuda::math::kernel {

DH_INLINE float gaussian(float distance, float sigma) {
    float a = distance / sigma;
    // return expf(-0.5f * a * a);
    return core::cuda::math::fast::expf(-0.5f * a * a);
}




} // namespace core::cuda::math::kernel

namespace TEMPLATE_NAMESPACE {

// add to math::kernel
// math::kernel::gaussian

// kernel in this context
// A radially symmetric weighting function that maps distance → influence.

// math::kernel::gaussian
// math::kernel::cubic_spline_sph
// math::kernel::wendland_c2

// ================================================================

// gaussian distance based falloff, will be 1 at distance 0
// sigma is radius
// sigma = radius * 0.5f; // tighter
// sigma = radius; // softer
DH_INLINE float gaussian_falloff(float distance, float sigma) {
    float a = distance / sigma;
    return expf(-0.5f * a * a);
    // return cmath::fast::expf(-0.5f * a * a);
}
// ----------------------------------------------------------------

// Wendland C2 radial basis function (compactly supported).
//
// d = distance from the evaluation point to the center.
// R = support radius. The kernel is guaranteed to be zero for d >= R.
//
// This kernel behaves similarly to a Gaussian near the center but has a
// hard cutoff at R, making it ideal for simulations where influence must
// remain local (e.g., SPH fluids, erosion, smoothing).
//
// Properties:
//   - C2 continuous (smooth second derivative)
//   - Compact support (finite radius)
//   - Monotonic, positive, bell-shaped
//   - Cheap to evaluate (pure polynomials, no exp())
//
// Formula:
//   φ(d) = (1 - d/R)^4 * (4(d/R) + 1), for d < R
//   φ(d) = 0, otherwise
DH_INLINE float wendland_c2(float d, float R) {
    if (d >= R) return 0.0f;
    float x = 1.0f - d / R;
    float x2 = x * x;
    return x2 * x2 * (4.0f * d / R + 1.0f);
}

// ================================================================

// Cubic spline SPH kernel (compact support).
// d = distance from sample to center.
// h = smoothing radius. Kernel is zero for d >= 2h.
//
// This is the standard SPH "workhorse" kernel:
//   - Bell-shaped, similar to a Gaussian near the center
//   - Smooth (C2 continuous)
//   - Compact support (finite radius)
//   - Cheap polynomial evaluation
//
// Normalization constant omitted since you're just visualizing.
DH_INLINE float cubic_spline(float d, float h) {
    float q = d / h;
    if (q >= 2.0f) return 0.0f;

    if (q < 1.0f) {
        return 1.0f - 1.5f * q * q + 0.75f * q * q * q;
    } else {
        float t = 2.0f - q;
        return 0.25f * t * t * t;
    }
}

DH_INLINE float linear_falloff(float d, float R) {
    float t = 1.0f - d / R;
    return t > 0.0f ? t : 0.0f;
}

DH_INLINE float hard_circle(float d, float R) {
    return d <= R ? 1.0f : 0.0f;
}

// ================================================================

__global__ void draw_kernel(
    int mode,
    int2 size,
    float *__restrict__ output,
    float2 position,
    float radius) {
    // ================================================================
    int2 pos = cmath::global_thread_pos2();
    if (pos.x >= size.x || pos.y >= size.y) return;
    int idx = cmath::pos_to_idx(pos, size);
    // ================================================================

    // Convert pixel to float2
    float2 p = make_float2((float)pos.x, (float)pos.y);

    // Distance to kernel center
    float2 d2 = make_float2(p.x - position.x, p.y - position.y);
    float d = sqrtf(d2.x * d2.x + d2.y * d2.y);

    // Evaluate cubic spline kernel

    float w;

    switch (mode) {
    case 0:
        w = cubic_spline(d, radius / 2.0f);
        break;
    case 1:
        w = wendland_c2(d, radius);
        break;
    case 2:

        w = cmath::kernel::gaussian(d, radius / 6.0f);

        break;
    case 3:
        w = linear_falloff(d, radius);
        break;
    case 4:
        w = hard_circle(d, radius);
        break;
    }

    // Write to output
    output[idx] = w;
}

void TEMPLATE_CLASS_NAME::_compute() {

    std::array<size_t, 2> shape{(size_t)size.x, (size_t)size.y};
    ensure_array_ref_ready(output, shape);

    dim3 block(16, 16);
    auto grid = cmath::calculate_grid(size, block);

    draw_kernel<<<grid, block, 0, stream->get()>>>(
        mode,
        size,
        output->dev_ptr(),
        position,
        radius);
}

} // namespace TEMPLATE_NAMESPACE
