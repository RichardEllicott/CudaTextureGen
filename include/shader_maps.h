/*


this was ported from Godot


to be honest, so far it's gonna have problems interfacing with numpy


really we will need things like


[r0, g0, b0, r1, g1, b1, r2, g2, b2...

[x0, y0, z0, x1, y1, z1, x2, y2, z2...

will need to rewrite??



reinterpret_cast???

std::vector<Vector3> vec = { {1,2,3}, {4,5,6}, {7,8,9} };
float* raw = reinterpret_cast<float*>(vec.data());

it's fast but unsupported, i will try to work on a flat array


*/
#pragma once

#include <algorithm>
#include <algorithm> // for std::max
#include <array>
#include <cmath>
#include <cmath>   // for std::sqrt
#include <cstddef> // for size_t
#include <iostream>
#include <vector>

#include "core.h"

namespace shader_maps {

/*


// wrapping mod function
int posmod(int value, int mod);

// Helper to clamp coordinates or wrap around
int image_position_to_index(const Vector2i &pos, const Vector2i &size, bool wrap);

Vector3 normalize(const Vector3 &v);

// normal map generator using basic sobel filter similar to https://github.com/cpetry/NormalMap-Online/
std::vector<Vector3> generate_normal_map_array(const std::vector<float> &image, Vector2i image_size, float normal_scale, bool wrap);

// generate AO map based on height, simulates light falling in gaps
std::vector<float> generate_ao_map_array(const std::vector<float> &image, Vector2i image_size, float radius = 5.0f, float strength = 1.0f, bool wrap = false);

*/

#pragma region EXAMPLES

// Function that returns a dynamically allocated array of floats
float *generate_array(size_t size) {
    float *arr = new float[size];
    for (size_t i = 0; i < size; ++i) {
        arr[i] = static_cast<float>(i) * 1.5f; // Example values
    }
    return arr;
}

int example() {
    size_t size = 5;
    float *myArray = generate_array(size);

    // Use the array
    for (size_t i = 0; i < size; ++i) {
        std::cout << "myArray[" << i << "] = " << myArray[i] << "\n";
    }

    // Clean up
    delete[] myArray;
    return 0;
}

// generate vector
std::vector<float> generate_array2(size_t size) {
    std::vector<float> arr(size);
    for (size_t i = 0; i < size; ++i) {
        arr[i] = i * 2.0f;
    }
    return arr;
}

// turn raw array to vector
std::vector<float> to_vector(float *data, size_t size) {
    return std::vector<float>(data, data + size);
}

#pragma endregion

static int posmod(int value, int mod) {
    int result = value % mod;
    return result < 0 ? result + mod : result;
}

static int image_position_to_index(Vector2i position, Vector2i image_size, bool wrap) {
    if (wrap) {
        position.x = posmod(position.x, image_size.x);
        position.y = posmod(position.y, image_size.y);
    } else {
        position.x = std::clamp(position.x, 0, image_size.x - 1);
        position.y = std::clamp(position.y, 0, image_size.y - 1);
    }
    return position.x + position.y * image_size.x;
}

static Vector2i index_to_image_position(int index, Vector2i image_size) {
    return Vector2i(index % image_size.x, index / image_size.x);
}

inline std::vector<float> generate_normal_map_array(const std::vector<float> &image, Vector2i image_size, float normal_scale, bool wrap) {

    std::vector<float> normal_map;

    if (image_size.x * image_size.y != image.size()) {
        throw std::invalid_argument("Image size does not match the number of pixels in the height map.");
    }

    int array_size = image_size.x * image_size.y * 3;
    normal_map.resize(array_size);

    for (int y = 0; y < image_size.y; y++) {
        for (int x = 0; x < image_size.x; x++) {

            /*
            // clockwise from top
            float top = image[image_position_to_index(Vector2i(x, y - 1), image_size, wrap)];
            float top_right = image[image_position_to_index(Vector2i(x + 1, y - 1), image_size, wrap)];
            float right = image[image_position_to_index(Vector2i(x + 1, y), image_size, wrap)];
            float bottom_right = image[image_position_to_index(Vector2i(x + 1, y + 1), image_size, wrap)];
            float bottom = image[image_position_to_index(Vector2i(x, y + 1), image_size, wrap)];
            float bottom_left = image[image_position_to_index(Vector2i(x - 1, y + 1), image_size, wrap)];
            float left = image[image_position_to_index(Vector2i(x - 1, y), image_size, wrap)];
            float top_left = image[image_position_to_index(Vector2i(x - 1, y - 1), image_size, wrap)];

            // // Apply the Sobel operator for X and Y gradients.
            float partialDerivativeX = (top_right + 2 * right + bottom_right) - (top_left + 2 * left + bottom_left);
            float partialDerivativeY = (bottom_left + 2 * bottom + bottom_right) - (top_left + 2 * top + top_right);
            */

            // NEW REFACTOR
            const Vector2i offsets[8] = {
                {0, -1}, // top
                {1, -1}, // top-right
                {1, 0},  // right
                {1, 1},  // bottom-right
                {0, 1},  // bottom
                {-1, 1}, // bottom-left
                {-1, 0}, // left
                {-1, -1} // top-left
            };

            float samples[8];
            for (int i = 0; i < 8; ++i) {
                Vector2i pos = Vector2i(x, y) + offsets[i];
                samples[i] = image[image_position_to_index(pos, image_size, wrap)];
            }

            // Apply Sobel operator
            float partialDerivativeX = (samples[1] + 2 * samples[2] + samples[3]) - (samples[7] + 2 * samples[6] + samples[5]);
            float partialDerivativeY = (samples[5] + 2 * samples[4] + samples[3]) - (samples[7] + 2 * samples[0] + samples[1]);

            // get the normal
            Vector3 normal(partialDerivativeX * normal_scale, -partialDerivativeY * normal_scale, 1.0f);
            normal = normal.normalized();

            // Convert the normal to color space ([-1, 1] -> [0, 1])
            Color color(
                0.5f + 0.5f * normal.x, // Red
                0.5f + 0.5f * normal.y, // Green
                0.5f + 0.5f * normal.z  // Blue
            );

            int base_index = image_position_to_index(Vector2i(x, y), image_size, wrap) * 3;
            normal_map[base_index] = color.r;
            normal_map[base_index + 1] = color.g;
            normal_map[base_index + 2] = color.b;
        }
    }

    return normal_map;
}

inline std::vector<float> generate_ao_map_array_OLD(const std::vector<float> &image, Vector2i image_size, int radius, bool wrap) {
    std::vector<float> ao_map;

    // Validate input size
    if (image.size() != image_size.x * image_size.y) {
        // UtilityFunctions::push_error("Input image size does not match the specified image dimensions.");
        return ao_map;
    }

    ao_map.resize(image.size());

    for (int y = 0; y < image_size.y; ++y) {
        for (int x = 0; x < image_size.x; ++x) {
            const int base_index = image_position_to_index(Vector2i(x, y), image_size, wrap);
            const float base_height = image[base_index];

            float occlusion = 0.0f;
            float total_weight = 0.0f;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0)
                        continue;

                    // Apply circular mask
                    if (dx * dx + dy * dy > radius * radius)
                        continue;

                    const Vector2i sample_pos(x + dx, y + dy);
                    const int sample_index = image_position_to_index(sample_pos, image_size, wrap);
                    const float neighbor_height = image[sample_index];

                    const float diff = neighbor_height - base_height;

                    // const float distance = MAX(sqrt(dx * dx + dy * dy), 1e-5f);
                    const float distance = std::max(static_cast<float>(std::hypot(dx, dy)), 1e-5f);

                    const float weight = 1.0f / (distance * distance + 1.0f);

                    // Optional: directional bias (e.g., emphasize occlusion from above)
                    // if (dy < 0) weight *= 1.5f;

                    if (diff > 0.0f) {
                        occlusion += diff * weight;
                    }
                    total_weight += weight;
                }
            }

            const float ao_value = std::clamp(1.0f - (occlusion / total_weight), 0.0f, 1.0f);
            ao_map[base_index] = ao_value;
        }
    }

    return ao_map;
}

inline std::vector<float> generate_ao_map(const std::vector<float> &image, Vector2i image_size, int radius, bool wrap) {
    std::vector<float> ao_map;

    // Validate input size: ensure the flat image buffer matches the declared dimensions
    if (image.size() != image_size.x * image_size.y) {
        throw std::invalid_argument(
            "generate_ao_map_array: image size mismatch — expected " +
            std::to_string(image_size.x * image_size.y) +
            " pixels, but got " + std::to_string(image.size()));
    }

    // Allocate output buffer (same size as input, one float per pixel)
    ao_map.resize(image.size());

    // Iterate over each pixel in the image
    for (int y = 0; y < image_size.y; ++y) {
        for (int x = 0; x < image_size.x; ++x) {
            const int base_index = image_position_to_index(Vector2i(x, y), image_size, wrap);
            const float base_height = image[base_index];

            float occlusion = 0.0f;
            float total_weight = 0.0f;

            // Sample surrounding pixels within circular radius
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0)
                        continue; // Skip self

                    // Apply circular mask (avoid square sampling)
                    if (dx * dx + dy * dy > radius * radius)
                        continue;

                    const Vector2i sample_pos(x + dx, y + dy);
                    const int sample_index = image_position_to_index(sample_pos, image_size, wrap);
                    const float neighbor_height = image[sample_index];

                    const float diff = neighbor_height - base_height;

                    // Compute distance with stability clamp
                    const float distance = std::max(static_cast<float>(std::hypot(dx, dy)), 1e-5f);

                    // Weight based on inverse square falloff
                    float weight = 1.0f / (distance * distance + 1.0f);

                    // Optional directional bias (e.g., emphasize occlusion from above)
                    // if (dy < 0) weight *= 1.5f;

                    // Accumulate occlusion only from higher neighbors
                    if (diff > 0.0f) {
                        occlusion += diff * weight;
                    }
                    total_weight += weight;
                }
            }

            // Normalize and invert occlusion to get AO value
            const float ao_value = std::clamp(1.0f - (occlusion / total_weight), 0.0f, 1.0f);
            ao_map[base_index] = ao_value;
        }
    }

    return ao_map;
}

} // namespace shader_maps