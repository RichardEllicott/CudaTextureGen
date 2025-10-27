/*

JUST JUNK ABOUT SOBEL FILTER

*/
#pragma once


#pragma region NOTES

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
