/*


this is a bit simple, maybe don't need it?



import numpy as np

def feather_edges(img, margin=32):
    h, w = img.shape
    out = img.copy()

    # Horizontal blend
    for i in range(margin):
        alpha = i / margin
        out[:, i] = (1 - alpha) * img[:, i] + alpha * img[:, w - margin + i]
        out[:, w - i - 1] = (1 - alpha) * img[:, w - i - 1] + alpha * img[:, margin - i - 1]

    # Vertical blend
    for j in range(margin):
        alpha = j / margin
        out[j, :] = (1 - alpha) * out[j, :] + alpha * out[h - margin + j, :]
        out[h - j - 1, :] = (1 - alpha) * out[h - j - 1, :] + alpha * out[margin - j - 1, :]

    return out




*/
#pragma once

namespace edge_feather {

__device__ float lerp(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

__global__ void feather_edges(
    const float *img, float *out,
    int width, int height, int margin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    float pixel = img[idx];

    float blend_x = 0.0f;
    float blend_y = 0.0f;

    // Horizontal feathering
    if (x < margin) {
        blend_x = float(x) / margin;
        int wrap_x = width - margin + x;
        pixel = lerp(pixel, img[y * width + wrap_x], blend_x);
    } else if (x >= width - margin) {
        blend_x = float(width - x - 1) / margin;
        int wrap_x = x - width + margin;
        pixel = lerp(pixel, img[y * width + wrap_x], blend_x);
    }

    // Vertical feathering
    if (y < margin) {
        blend_y = float(y) / margin;
        int wrap_y = height - margin + y;
        pixel = lerp(pixel, img[wrap_y * width + x], blend_y);
    } else if (y >= height - margin) {
        blend_y = float(height - y - 1) / margin;
        int wrap_y = y - height + margin;
        pixel = lerp(pixel, img[wrap_y * width + x], blend_y);
    }

    out[idx] = pixel;
}

// TEST??
// dim3 blockSize(16, 16);
// dim3 gridSize((width + 15) / 16, (height + 15) / 16);
// feather_edges<<<gridSize, blockSize>>>(img, out, width, height, margin);


} // namespace edge_feather
