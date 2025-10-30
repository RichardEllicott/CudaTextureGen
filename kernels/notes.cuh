/*

JUST JUNK ABOUT SOBEL FILTER

*/
#pragma once


#pragma region NOTES




// 🚧🚧🚧🚧🚧 UNTESTED AI GENERATED 🚧🚧🚧🚧🚧
//
// Apply inverse-square crater imprint onto a heightmap.
// h: heightmap (row-major), W,H: dimensions
// cx,cy: impact center in pixels (float for subpixel)
// k: excavation scale (meters per unit energy)
// r0: softening radius in pixels (prevents singularities)
// mask_radius: optional clamp for finite blast radius
__global__ void crater_imprint(float *h, int W, int H,
                               float cx, float cy,
                               float k, float r0, float mask_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H)
        return;

    float dx = (x + 0.5f) - cx;
    float dy = (y + 0.5f) - cy;
    float r2 = dx * dx + dy * dy;

    if (mask_radius > 0.0f && r2 > mask_radius * mask_radius)
        return;

    float denom = r2 + r0 * r0; // softening
    float E = 1.0f / denom;     // inverse-square
    float dh = -k * E;          // excavation depth

    // Optional: taper center to avoid a pixel spike when r0 is small
    // dh *= (r2 / (r2 + r0*r0));

    int idx = y * W + x;
    h[idx] += dh;
}





// 🚧 my personal design
__global__ void notes(
    // Parameters *pars,
    int width, int height,
    float *height_map, float *sediment_map, float *water_map) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

// downhill slope calculation
#if 0
    int idx = y * width + x;
    float h = height_map[idx];

    // Offsets, axis first, then diagonals
    const int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

    float dir_x = 0.0f; // flow direction x
    float dir_y = 0.0f;

    for (int i = 0; i < 8; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height)
            continue;

        int nIdx = ny * width + nx;
        float nh = height_map[nIdx];
        float slope = h - nh;

        if (slope > 0.0f) {
            dir_x += dx[i] * slope;
            dir_y += dy[i] * slope;
        }
    }
#endif

// sobel pattern
#if 0 
    // Offsets clockwise from top
    const int ox[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    const int oy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

    float samples[8];
    for (int i = 0; i < 8; ++i) {
        int nx = x + ox[i];
        int ny = y + oy[i];
        int idx = image_position_to_index(nx, ny, width, height, true);
        samples[i] = height_map[idx];
    }

    // Sobel operator
    float dx = (samples[1] + 2 * samples[2] + samples[3]) - (samples[7] + 2 * samples[6] + samples[5]);
    float dy = (samples[5] + 2 * samples[4] + samples[3]) - (samples[7] + 2 * samples[0] + samples[1]);
#endif
}

#pragma endregion
