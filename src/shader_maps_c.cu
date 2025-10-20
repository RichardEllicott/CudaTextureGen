#include "shader_maps_c.cuh"

namespace shader_maps_c {

// Addressing helper
__device__ int image_position_to_index(int x, int y, const int width, const int height, const bool wrap) {
    if (wrap) {
        x = (x % width + width) % width;
        y = (y % height + height) % height;
    } else {
        x = min(max(x, 0), width - 1);
        y = min(max(y, 0), height - 1);
    }
    return y * width + x;
}

// Normalize a 3D vector
__device__ void normalize3(float &x, float &y, float &z) {
    float len = sqrtf(x * x + y * y + z * z);
    if (len > 1e-6f) {
        x /= len;
        y /= len;
        z /= len;
    }
}

__global__ void generate_normal_map_kernel(const float *__restrict__ heightmap,
                                           float *__restrict__ normalmap,
                                           int width, int height,
                                           float normal_scale, bool wrap) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Offsets clockwise from top
    const int ox[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    const int oy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

    float samples[8];
    for (int i = 0; i < 8; ++i) {
        int nx = x + ox[i];
        int ny = y + oy[i];
        int idx = image_position_to_index(nx, ny, width, height, wrap);
        samples[i] = heightmap[idx];
    }

    // Sobel operator
    float dx = (samples[1] + 2 * samples[2] + samples[3]) -
               (samples[7] + 2 * samples[6] + samples[5]);
    float dy = (samples[5] + 2 * samples[4] + samples[3]) -
               (samples[7] + 2 * samples[0] + samples[1]);

    float nx = dx * normal_scale;
    float ny = -dy * normal_scale;
    float nz = 1.0f;

    normalize3(nx, ny, nz);

    // Convert to [0,1] color space
    float r = 0.5f + 0.5f * nx;
    float g = 0.5f + 0.5f * ny;
    float b = 0.5f + 0.5f * nz;

    int base = (y * width + x) * 3;
    normalmap[base + 0] = r;
    normalmap[base + 1] = g;
    normalmap[base + 2] = b;
}

// host_in: pointer to heightmap data (width*height floats)
// host_out: pointer to output normal map (width*height*3 floats)
void ShaderMaps::generate_normal_map(
    const float *host_in, float *host_out,
    int width, int height,
    float normal_scale, bool wrap) {

    size_t in_size = width * height * sizeof(float);
    size_t out_size = width * height * 3 * sizeof(float);

    float *d_in = nullptr;
    float *d_out = nullptr;

    cudaMalloc(&d_in, in_size);
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_in, host_in, in_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    generate_normal_map_kernel<<<grid, block>>>(
        d_in, d_out, width, height, normal_scale, wrap);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(host_out, d_out, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

} // namespace shader_maps_c