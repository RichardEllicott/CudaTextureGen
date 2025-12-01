/*

store some kernels

*/
// #pragma once

namespace TEMPLATE_NAMESPACE {

#pragma region HASH

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

// Hash returning float in [0,1)
__device__ __forceinline__ float hash_float(int x, int y, int z, int seed) {
    int h = hash_int(x, y, z, seed);
    // Scale to [0,1). Use 1.0f / 2147483648.0f (2^31)
    return static_cast<float>(h) * (1.0f / 2147483648.0f);
}

// If you want [-1,1] range:
__device__ __forceinline__ float hash_float_signed(int x, int y, int z, int seed) {
    int h = hash_int(x, y, z, seed);
    return static_cast<float>(h) * (2.0f / 2147483648.0f) - 1.0f;
}

#pragma endregion

#pragma region HELPERS

constexpr float SQRT2 = 1.4142135623730950488f;

// 8 offsets with the opposites in pairs
__device__ __constant__ int2 offsets[8] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
__device__ __constant__ float offset_distances[8] = {1.0f, 1.0f, 1.0f, 1.0f, SQRT2, SQRT2, SQRT2, SQRT2};
__device__ __constant__ int opposite_offset_refs[8] = {1, 0, 3, 2, 5, 4, 7, 6};

// tested posmod for wrapping map coordinates
__device__ __forceinline__ int posmod(int i, int mod) {
    int result = i % mod;
    return result < 0 ? result + mod : result;
}

// wrap or clamp for map coordinates, note the clamp is range-1
__device__ __forceinline__ int wrap_or_clamp(int i, int range, bool wrap) {
    if (wrap) {
        return posmod(i, range);
    } else {
        return i < 0 ? 0 : (i >= range ? range - 1 : i);
    }
}

__device__ __forceinline__ int clampi(int value, int minimum, int maximum) {
    return min(max(value, minimum), maximum);
}

__device__ __forceinline__ float clampf(float value, float minimum, float maximum) {
    return fminf(fmaxf(value, minimum), maximum);
}

#pragma endregion

#pragma region KERNELS

__global__ void rain_pass(
    const Parameters *pars,
    const int map_width, const int map_height,
    curandState *rand_states,

    const float *__restrict__ rain_map, // 🚧 optional in

    float *__restrict__ water_map // out

) {
    // ================================================================
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height)
        return;
    int idx = y * map_width + x;
    // ================================================================

    float rain = pars->rain_rate;

    if (rain_map) {
        rain *= rain_map[idx]; // multiply by rain_map if != nullptr
    }

    if (pars->rain_random) {
        // generate a random number r
        curandState local = rand_states[idx];
        float r = curand_uniform(&local); // (0,1]
        rand_states[idx] = local;         // save updated state
        rain *= r;
    }

    water_map[idx] += rain;
}

__global__ void calculate_flux(
    const Parameters *pars,
    const int map_width, const int map_height,
    const int step,

    const float *__restrict__ height_map,   // in
    const float *__restrict__ water_map,    // in
    const float *__restrict__ sediment_map, // in

    float *__restrict__ flux8,          // out
    float *__restrict__ sediment_flux8, // out
    float *__restrict__ slope_map       // out
) {
    // ================================================================
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height)
        return;
    int idx = y * map_width + x;
    // ================================================================

    float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];
    float surface = height + water;

    // ================================================================
    // [Calculate Slopes]
    // ----------------------------------------------------------------
    float slopes[8];        // positive slope to neighbours
    float sum_slope = 0.0f; // total slope

    for (int n = 0; n < 8; ++n) {

        // new pos
        int nx = wrap_or_clamp(x + offsets[n].x, map_width, pars->wrap);
        int ny = wrap_or_clamp(y + offsets[n].y, map_height, pars->wrap);
        int nidx = ny * map_width + nx;

        float n_surface = height_map[nidx] + water_map[nidx]; // new surface
        float difference = surface - n_surface;               // positive means we are higher

        if (pars->slope_jitter > 0.0f) {
            // float jitter = pars->slope_jitter * noise_util::trig_hash(x, y, n + 1234, step);
            float jitter = pars->slope_jitter * hash_float_signed(x, y, n, step);
            difference += jitter;
        }

        if (pars->correct_diagonal_distance) {
            difference /= offset_distances[n];
        }

        // amount higher than we are than neighbour (but 0.0 if difference is below slope_threshold)
        float positive_difference = difference > pars->slope_threshold ? difference : 0.0f; // amount higher we are than neighbour (or 0 if we are lower)

        slopes[n] = positive_difference;
        sum_slope += positive_difference;
    }

    slope_map[idx] = sum_slope; // OUT

    // ================================================================
    // [Calculate Flux]
    // ----------------------------------------------------------------
    float max_outflow = fminf(water, pars->max_water_outflow); // slows outflow if above threshold
    float *cell_flux = &flux8[idx * 8];                        // pointer to this cell’s 8‑flux slice

    if (sum_slope > 1e-6f && max_outflow > 0.f) {
        for (int n = 0; n < 8; ++n) {
            float q = (slopes[n] / sum_slope) * max_outflow; // proportional share of outflow
            cell_flux[n] = q;
        }
    } else {
        for (int n = 0; n < 8; ++n) {
            cell_flux[n] = 0.f; // no downhill slope or no water: zero flux
        }
    }

    // ================================================================
    // [Isotropic Diffusion]
    // ----------------------------------------------------------------
    if (pars->diffusion_rate > 0.0f) {

        for (int n = 0; n < 8; ++n) {
            int nx = wrap_or_clamp(x + offsets[n].x, map_width, pars->wrap);
            int ny = wrap_or_clamp(y + offsets[n].y, map_height, pars->wrap);
            int nidx = ny * map_width + nx;

            float neighbor_water = water_map[nidx];
            float delta = neighbor_water - water; // positive if neighbor has more
            float diffusion_flux = pars->diffusion_rate * delta;

            cell_flux[n] += diffusion_flux;
        }
    }

    // ================================================================
    // [Sediment Flux]
    // ----------------------------------------------------------------
    float sediment_concentration = (water > 1e-6f) ? (sediment / water) : 0.0f;
    for (int n = 0; n < 8; ++n) {
        float q = flux8[idx * 8 + n];                             // water flux to neighbor
        sediment_flux8[idx * 8 + n] = sediment_concentration * q; // sediment carried with that water
    }
}

__global__ void apply_flux(
    const Parameters *pars,
    const int map_width, const int map_height,

    const float *__restrict__ height_map,     // in
    const float *__restrict__ water_map,      // in
    const float *__restrict__ sediment_map,   // in
    const float *__restrict__ flux8,          // in
    const float *__restrict__ sediment_flux8, // in
    const float *__restrict__ slope_map,      // in

    const float *__restrict__ hardness_map, // 🚧 optional in

    float *__restrict__ height_map_out,  // out
    float *__restrict__ water_map_out,   // out
    float *__restrict__ sediment_map_out // out

) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= map_width || y >= map_height)
        return;
    int idx = y * map_width + x;
    // values change in this pass and are written out
    float height = height_map[idx];
    float water = water_map[idx];
    float sediment = sediment_map[idx];
    float slope = slope_map[idx]; // the strength of the slope from previous pass

    // ================================================================
    // [Calculate and Apply Flow]
    // ----------------------------------------------------------------
    float water_inflow = 0.f;
    float water_outflow = 0.0f;

    for (int n = 0; n < 8; ++n) {
        // calculate outflow
        water_outflow += flux8[idx * 8 + n];
        // calculate inflow
        int nx = wrap_or_clamp(x + offsets[n].x, map_width, pars->wrap);
        int ny = wrap_or_clamp(y + offsets[n].y, map_height, pars->wrap);
        int nidx = ny * map_width + nx;

        int opposite_offset = opposite_offset_refs[n];
        water_inflow += flux8[nidx * 8 + opposite_offset];
    }
    water -= water_outflow;
    water += water_inflow;

    // ================================================================
    // [Carving (basic optional)]
    // ----------------------------------------------------------------
    height -= water_outflow * pars->outflow_carve;

    // ================================================================
    // [Simple Erode (like previous model)]
    // ----------------------------------------------------------------
    height -= slope * pars->simple_erosion_rate;

    // ================================================================
    // [Erosion]
    // ----------------------------------------------------------------
    float erosion = water_outflow * pars->erosion_rate; // erosion just based on outflow * the erosion rate

    // in the case of erosion mode's 1-3 the slope also affects the erosion
    switch (pars->erosion_mode) {
    case 1: // slope_factor only
        erosion *= slope;
        break;
    case 2: // soft saturate slope_factor
        erosion *= slope / (1.0f + slope);
        break;
    case 3: // exponent based slope_factor
        erosion *= powf(slope, pars->slope_exponent);
        break;
    }

    float available_erosion = height - pars->min_height; // limit erosion to available rock above min_height
    erosion = fminf(erosion, fmaxf(0.0f, available_erosion));
    sediment += erosion;
    height -= erosion;

    height = clampf(height, pars->min_height, pars->max_height); // clamp height (but min is already enforced)

    // ================================================================
    // [Evaporation]
    // ----------------------------------------------------------------
    water -= pars->evaporation_rate;

    // ================================================================
    // [Sediment Transport]
    // ----------------------------------------------------------------
    float sediment_change = 0.0f;
    for (int n = 0; n < 8; ++n) {
        int nx = wrap_or_clamp(x + offsets[n].x, map_width, pars->wrap);
        int ny = wrap_or_clamp(y + offsets[n].y, map_height, pars->wrap);
        int nidx = ny * map_width + nx;
        int opp = opposite_offset_refs[n];

        sediment_change -= sediment_flux8[idx * 8 + n];    // outflow
        sediment_change += sediment_flux8[nidx * 8 + opp]; // inflow
    }
    sediment += sediment_change;

    // ================================================================
    // [Deposition]
    // ----------------------------------------------------------------
    if (pars->deposition_mode == 0) {
        // basic deposition
        float deposit = sediment * pars->deposition_rate;
        sediment -= deposit;
        height += deposit;
    } else if (pars->deposition_mode == 1) {
        // capacity based
        float capacity = pars->sediment_capacity * water * slope;
        if (sediment > capacity) {
            float deposit = sediment_map[idx] - capacity;
            sediment -= deposit;
            height += deposit;
        }
    }

    if (pars->drain_at_min_height && height <= pars->min_height) {
        water = 0.0f;
        sediment = 0.0f;
    }

    // ================================================================
    height_map_out[idx] = height;
    water_map_out[idx] = fmaxf(0.f, water);       // no negative water
    sediment_map_out[idx] = fmaxf(0.f, sediment); // no negative sediment
}

#pragma endregion

} // namespace TEMPLATE_NAMESPACE
