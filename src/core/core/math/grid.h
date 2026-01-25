/*

grid functions

*/
#pragma once

#include <cstdint> // uint32_t
#include <vector>

#include "core/cuda_compat.h"

namespace core::math::grid {

#pragma region SQUARE_RING

// quater of a square perimeter ring (in order)
inline std::vector<int2> square_ring_quart(int radius = 1) {
    std::vector<int2> out;

    for (int i = 0; i < radius * 2; i++)
        out.push_back({-radius + 1 + i, radius}); // gives the seg from (-n+1,n) → (n,n)
    return out;
}

// half of a square perimeter ring (in order)
inline std::vector<int2> square_ring_half(int radius = 1) {
    auto in = square_ring_quart(radius);
    std::vector<int2> out = in;
    out.reserve(in.size() * 2);

    for (const auto &pos : in)
        out.push_back({-pos.y, pos.x}); // 90° rotated point

    return out;
}

// square perimeter ring (in order)
inline std::vector<int2> square_ring(int radius = 1) {
    auto in = square_ring_half(radius);
    std::vector<int2> out = in;
    out.reserve(in.size() * 2);

    for (const auto &pos : in)
        out.push_back({-pos.x, -pos.y}); // 180° rotated point

    return out;
}

// square perimeter ring (interlaced: point, opposite, point, opposite...)
// this way ref i ^ 1 is the opposite
inline std::vector<int2> square_ring_interlaced(int radius = 1) {
    auto in = square_ring_half(radius);
    std::vector<int2> out;
    out.reserve(in.size() * 2);

    for (const auto &pos : in) {
        out.push_back(pos);              // original
        out.push_back({-pos.x, -pos.y}); // opposite (180° rotation)
    }

    return out;
}

// ================================================================

std::vector<int2> get_surrounding_offsets(int order, int mode = 0) {

    std::vector<int2> result;

    for (int r = 1; r <= order; r++) {
        std::vector<int2> ring;

        switch (mode) {
        case 0:
            ring = math::grid::square_ring(r);
            break;
        case 1:
            ring = math::grid::square_ring_half(r);
            break;
        case 2:
            ring = math::grid::square_ring_quart(r);

            break;
        }

        result.insert(result.end(), ring.begin(), ring.end());
    }

    return result;
}

#pragma endregion

std::vector<int2> filter_by_distance(
    const std::vector<int2> &tiles,
    float radius) {

    std::vector<int2> result;
    if (tiles.empty())
        return result;

    // Compute bounding box to find center
    int min_x = tiles[0].x, max_x = tiles[0].x;
    int min_y = tiles[0].y, max_y = tiles[0].y;

    for (const auto &t : tiles) {
        min_x = std::min(min_x, t.x);
        max_x = std::max(max_x, t.x);
        min_y = std::min(min_y, t.y);
        max_y = std::max(max_y, t.y);
    }

    // Center of the tile set (float for accuracy)
    float cx = (min_x + max_x) * 0.5f;
    float cy = (min_y + max_y) * 0.5f;

    float r2 = radius * radius;

    // Filter tiles inside the radius
    for (const auto &t : tiles) {
        float dx = float(t.x) - cx;
        float dy = float(t.y) - cy;
        float d2 = dx * dx + dy * dy;

        if (d2 <= r2)
            result.push_back(t);
    }

    return result;
}

H_INLINE std::string tiles_to_string(
    const std::vector<int2> &tiles,
    const std::string &filled = "[]",
    const std::string &empty = "  ") {
    if (tiles.empty())
        return {};

    // Compute bounding box
    int min_x = tiles[0].x, max_x = tiles[0].x;
    int min_y = tiles[0].y, max_y = tiles[0].y;

    for (const auto &t : tiles) {
        min_x = std::min(min_x, t.x);
        max_x = std::max(max_x, t.x);
        min_y = std::min(min_y, t.y);
        max_y = std::max(max_y, t.y);
    }

    const int width = max_x - min_x + 1;
    const int height = max_y - min_y + 1;

    // Boolean mask
    std::vector<uint8_t> mask(width * height, 0);
    for (const auto &t : tiles) {
        int gx = t.x - min_x;
        int gy = t.y - min_y;
        mask[gy * width + gx] = 1;
    }

    // Build output
    std::string out;
    out.reserve((width * filled.size() + 1) * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            out += (mask[y * width + x] ? filled : empty);
        }
        out += '\n';
    }

    return out;
}

} // namespace core::math::grid
