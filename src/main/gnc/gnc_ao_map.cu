#include "gnc/gnc_ao_map.cuh"

#include "core/cuda/math.cuh"

namespace TEMPLATE_NAMESPACE {

#pragma region AO_MAP

// Addressing helper
__device__ __forceinline__ int image_position_to_index(int x, int y, const int width, const int height, const bool wrap) {
    if (wrap) {
        x = (x % width + width) % width;
        y = (y % height + height) % height;
    } else {
        x = min(max(x, 0), width - 1);
        y = min(max(y, 0), height - 1);
    }
    return y * width + x;
}

// RTAO = Relative/Regular Terrain Ambient Occlusion (sometimes called “fast AO” in Designer docs)
// simply check radius
__global__ void rtao_map_kernel(
    const float *__restrict__ image, float *__restrict__ ao_map,
    int width, int height,
    int radius, bool wrap) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const int base_index = y * width + x;
    const float base_height = image[base_index];

    float occlusion = 0.0f;
    float total_weight = 0.0f;

    // Accumulate within circular radius
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0)
                continue;
            if (dx * dx + dy * dy > radius * radius)
                continue;

            const int sample_index = image_position_to_index(x + dx, y + dy, width, height, wrap);
            const float neighbor_height = image[sample_index];

            const float diff = neighbor_height - base_height;

            // Distance with stability clamp
            const float dist2 = static_cast<float>(dx * dx + dy * dy);
            const float distance = fmaxf(sqrtf(dist2), 1e-5f);

            // Inverse-square falloff (with +1 to avoid singularity)
            float weight = 1.0f / (distance * distance + 1.0f);

            if (diff > 0.0f) {
                occlusion += diff * weight;
            }
            total_weight += weight;
        }
    }

    // Normalize and invert to get AO
    float ao_value = 1.0f - (occlusion / fmaxf(total_weight, 1e-8f));
    // Clamp to [0,1]
    ao_value = fminf(fmaxf(ao_value, 0.0f), 1.0f);

    ao_map[base_index] = ao_value;
}

// Direction set: fixed unit vectors, optionally in constant memory
// Keep N small (e.g., 8–16) for performance; jitter if you want to break patterns.
__device__ __forceinline__ float2 dir_from_index(int k, int N) {
    float angle = (2.0f * 3.14159265358979323846f) * (float(k) / float(N));
    return make_float2(cosf(angle), sinf(angle));
}

// HTAO kernel
//
// Fast preview: directions=6, steps_per_dir=6, max_radius=12, step_size=1.0, alpha=0.7, k=2.0 → Quick, coarse AO, good for iteration.
// Balanced quality: directions=8, steps_per_dir=12, max_radius=16, step_size=1.0, alpha=0.7, k=2.0 → Smooth enough for most uses, still performant.
// High quality bake: directions=12, steps_per_dir=16, max_radius=24, step_size=0.75, alpha=0.8, k=2.5 → Very smooth, captures both fine crevices and broad occlusion, but heavier.

struct HTAO_Pars {
    int directions;
    int steps_per_dir;
    int max_radius;
    float step_size;
    float alpha;
    float k;

    // Low quality / fast preview
    static HTAO_Pars Low() {
        return {
            6,    // directions
            6,    // steps_per_dir
            12,   // max_radius
            1.0f, // step_size
            0.7f, // alpha
            2.0f  // k
        };
    }

    // Balanced quality
    static HTAO_Pars Medium() {
        return {
            8,    // directions
            12,   // steps_per_dir
            16,   // max_radius
            1.0f, // step_size
            0.7f, // alpha
            2.0f  // k
        };
    }

    // High quality / final bake
    static HTAO_Pars High() {
        return {
            12,    // directions
            16,    // steps_per_dir
            24,    // max_radius
            0.75f, // step_size (sub-texel for smoother marching)
            0.8f,  // alpha (more directional)
            2.5f   // k (slightly lighter AO)
        };
    }
};

__global__ void
htao_kernel(
    const float *__restrict__ image,
    float *__restrict__ ao_map,
    const int width, const int height,

    // int max_radius,    // in texels
    // int directions,    // e.g. 12
    // int steps_per_dir, // march samples per direction (<= max_radius)
    // float step_size,   // sub-texel marching, e.g. 1.0f
    bool wrap,

    const HTAO_Pars *const __restrict__ pars

) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const int base_idx = y * width + x;
    const float base_h = image[base_idx];

    float occlusion_sum = 0.0f;

    // For each direction, march and find the maximum horizon slope
    for (int k = 0; k < pars->directions; ++k) {
        const float2 dir = dir_from_index(k, pars->directions);

        float max_slope = -1e30f; // track steepest rise relative to base
        float accum = 0.0f;       // optional integration of slope along path

        // March outward
        for (int s = 1; s <= pars->steps_per_dir; ++s) {
            float d = pars->step_size * float(s);
            if (d > float(pars->max_radius))
                break;

            // Sub-texel sampling coordinates
            float fx = float(x) + dir.x * d;
            float fy = float(y) + dir.y * d;

            // Nearest-neighbor; you can upgrade to bilinear for smoother HTAO
            int sx = int(fx + 0.5f);
            int sy = int(fy + 0.5f);
            int si = image_position_to_index(sx, sy, width, height, wrap);

            float h = image[si];

            // Slope relative to base over distance d
            float slope = (h - base_h) / (d + 1e-6f);
            max_slope = fmaxf(max_slope, slope);

            // Optional: accumulate clamped positive slope along the ray
            accum += fmaxf(slope, 0.0f);
        }

        // Convert horizon measure to occlusion contribution
        // Two options: (A) horizon-angle style, (B) integrated slope.
        // A: Use max_slope as proxy for horizon angle → more directional
        float dir_occ_A = fmaxf(max_slope, 0.0f);

        // B: Integrated slope along the ray → smoother but less directional
        float dir_occ_B = accum / (float)pars->steps_per_dir;

        // Blend them; tweak alpha in [0,1] for look
        float dir_occ = pars->alpha * dir_occ_A + (1.0f - pars->alpha) * dir_occ_B;

        // Nonlinear remap to keep values bounded and soft
        // k controls sensitivity to small slopes
        // const float k = 2.0f;
        dir_occ = dir_occ / (dir_occ + pars->k);

        occlusion_sum += dir_occ;
    }

    // Average across directions
    float occ = occlusion_sum / fmaxf(float(pars->directions), 1.0f);

    // AO is inverse of occlusion; clamp to [0,1]
    float ao = 1.0f - occ;
    ao = fminf(fmaxf(ao, 0.0f), 1.0f);

    ao_map[base_idx] = ao;
}

void generate_ao_map(
    const float *host_in, float *host_out,
    int width, int height,
    int radius, bool wrap, int mode) {

    size_t in_size = width * height * sizeof(float);
    size_t out_size = width * height * sizeof(float);

    float *d_in = nullptr;
    float *d_out = nullptr;

    cudaMalloc(&d_in, in_size);
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_in, host_in, in_size, cudaMemcpyHostToDevice);

    HTAO_Pars pars;

    switch (mode) {
    default:
        pars = HTAO_Pars::Low();
        break;
    case 2:
        pars = HTAO_Pars::Medium();
        break;
    case 3:
        pars = HTAO_Pars::High();
        break;
    }
    core::cuda::DeviceStruct<HTAO_Pars> _pars(pars);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    switch (mode) {
    case 0:
        rtao_map_kernel<<<grid, block>>>(
            d_in, d_out,
            width, height,
            radius, wrap);

        break;
    case 1:

        htao_kernel<<<grid, block>>>(
            d_in, d_out,
            width, height,
            wrap, _pars.dev_ptr());

        break;
    case 2:

        htao_kernel<<<grid, block>>>(
            d_in, d_out,
            width, height,
            wrap, _pars.dev_ptr());
        break;
    case 3:

        htao_kernel<<<grid, block>>>(
            d_in, d_out,
            width, height,
            wrap, _pars.dev_ptr());
        break;

    default:
        break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(host_out, d_out, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

#pragma endregion

void TEMPLATE_CLASS_NAME::_compute() {

    if (!input.is_valid()) throw std::runtime_error("input is not valid");
    if (input->empty()) throw std::runtime_error("input is empty");

    auto shape = input->shape();
    _size = to_int2(shape);
    // auto shape3 = std::array{shape[0], shape[1], (size_t)3}; // RGB size
    ensure_array_ref_ready(output, shape);

    dim3 block(16, 16);
    dim3 grid = cmath::calculate_grid(_size, block);

    // // ready_device();

    // normal_map_kernel<<<grid, block, 0, stream->get()>>>(
    //     input->dev_ptr(),
    //     output->dev_ptr(),
    //     _size,
    //     normal_scale,
    //     direct_x_style);
}

} // namespace TEMPLATE_NAMESPACE
