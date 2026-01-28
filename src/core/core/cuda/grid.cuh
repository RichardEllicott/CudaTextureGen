/*

*/
#pragma once

#include <string>
#include <vector>
#include <functional>


#include "core/cuda/math.cuh"
#include "core/defines.h"

namespace core::cuda::grid {

#pragma region KERNEL_BUILDING

H_INLINE std::string grid_positions_to_string(
    const std::vector<int2> &tiles,
    const std::string &filled = "[]",
    const std::string &empty = "__",
    bool with_axes = false) {
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

    const int cell_w = (int)empty.size();
    const int label_w = with_axes ? 4 : 0; // enough for negative coords

    out.reserve((width * filled.size() + label_w + 4) * height);

    // Rows (top → bottom)
    for (int y = 0; y < height; y++) {

        if (with_axes) {
            int actual_y = max_y - y; // real coordinate
            std::string s = std::to_string(actual_y);

            // right-align inside label_w
            out.append(label_w - (int)s.size(), ' ');
            out += s;
        }

        // Row contents
        for (int x = 0; x < width; x++) {
            out += (mask[y * width + x] ? filled : empty);
        }

        out += '\n';
    }

    // Bottom axis
    if (with_axes) {
        out.append(label_w, ' '); // left margin

        for (int x = 0; x < width; x++) {
            int actual_x = min_x + x;
            std::string s = std::to_string(actual_x);

            // center inside cell width
            int pad = cell_w - (int)s.size();
            int left = pad / 2;
            int right = pad - left;

            out.append(left, ' ');
            out += s;
            out.append(right, ' ');
        }

        out += '\n';
    }

    return out;
}

DH_INLINE int2 square_ring_quart_at(int radius = 1, int quart = 0, int i = 0) {
    switch (quart) {
    case 0: return {-radius + 1 + i, radius};  // top edge: (-r+1, r) → (r, r)
    case 1: return {-radius, -radius + 1 + i}; // left edge: (-r, -r+1) → (-r, r)
    case 2: return {radius - 1 - i, -radius};  // bottom edge: (r-1, -r) → (-r, -r)
    default: return {radius, radius - 1 - i};  // right edge: (r, r-1) → (r, -r)
    }
}

H_INLINE std::vector<int2> square_ring_quart(int radius = 1, int quart = 0) {
    std::vector<int2> out;
    int count = radius * 2;
    out.reserve(count);
    // quart &= 3; // cheap mod 4
    for (int i = 0; i < count; i++) {
        out.push_back(square_ring_quart_at(radius, quart, i));
    }
    return out;
}

H_INLINE std::vector<int2> surrounding_offsets_quart(int order, int quart = 0) {
    std::vector<int2> result;
    result.reserve(order * (order + 1));

    for (int radius = 1; radius <= order; radius++) {
        int count = radius * 2;
        for (int i = 0; i < count; i++)
            result.push_back(square_ring_quart_at(radius, quart, i));
    }
    return result;
}

// filter vector with predicate
template <class T, class Predicate>
std::vector<T> filter(const std::vector<T> &in, Predicate predicate) {
    std::vector<T> out;
    out.reserve(in.size());
    for (const auto &v : in)
        if (predicate(v))
            out.push_back(v);
    return out;
}

// filter vector by max length
H_INLINE std::vector<int2> filter_by_max_length(const std::vector<int2> &positions, float max_length) {
    return filter(
        positions,
        [&](int2 p) { return core::cuda::math::length(p) <= max_length; });
}

#pragma region SAMPLING_FIELD_GENERATOR

struct SamplingFieldGeneratorEntry {
    int2 offset;
    float distance;
    float weight;
    float2 direction;
    float2 dot_vector;

    SamplingFieldGeneratorEntry get_rotation(int quarter_turns) const {
        return {
            core::cuda::math::rotate(offset, quarter_turns),
            distance,
            weight,
            core::cuda::math::rotate(direction, quarter_turns),
            core::cuda::math::rotate(dot_vector, quarter_turns)};
    }
};

class SamplingFieldGenerator {

    using Entry = SamplingFieldGeneratorEntry;

    std::vector<Entry> entries;
    float total_weight = 0.0f;

    // float _distance_to_weight(float distance) {
    //     switch (weight_mode) {
    //     case 1:
    //         return core::cuda::math::kernel::gaussian(distance, 1.0f);
    //     default:
    //         return 1.0f / distance; // inverse-distance
    //     }
    // }

    void _starting_data() {

        using core::cuda::cast::to_float2;

        entries.clear();
        total_weight = 0.0f;

        auto offsets = surrounding_offsets_quart(radius);
        if (circular) offsets = filter_by_max_length(offsets, radius + circular_fraction);

        entries.reserve(offsets.size());

        for (int i = 0; i < offsets.size(); i++) {
            Entry entry;
            entry.offset = offsets[i];
            entry.distance = core::cuda::math::length(entry.offset);
            entry.weight = distance_formula(entry.distance);
            total_weight += entry.weight;
            entry.direction = core::cuda::math::normalize(to_float2(entry.offset));
            entries.push_back(entry);
        }
    }

    void _normalize_weights() {
        for (auto &entry : entries) {
            entry.weight /= total_weight;
            entry.dot_vector = entry.direction * entry.weight;
        }
        total_weight = 1.0f;
    }

    void _duplicate_as_rotation(int quarter_turns, bool interlace) {

        int count = entries.size();

        if (interlace) {
            std::vector<Entry> out;
            out.reserve(count * 2);

            for (int i = 0; i < count; i++) {
                const auto &entry = entries[i];
                out.push_back(entry);
                out.push_back(entry.get_rotation(quarter_turns));
            }

            entries = std::move(out);
        } else {
            entries.reserve(count * 2);

            for (int i = 0; i < count; i++) {
                entries.push_back(entries[i].get_rotation(quarter_turns));
            }
        }
    }

    template <typename T>
    std::vector<T> extract(T Entry::*member) const {
        std::vector<T> out;
        out.reserve(entries.size());
        for (const auto &e : entries) out.push_back(e.*member);
        return out;
    }

  public:
    std::vector<int2> get_offsets() const { return extract(&Entry::offset); }
    std::vector<float> get_distances() const { return extract(&Entry::distance); }
    std::vector<float> get_weights() const { return extract(&Entry::weight); }
    std::vector<float2> get_directions() const { return extract(&Entry::direction); }
    std::vector<float2> get_dot_vectors() const { return extract(&Entry::dot_vector); }

  public:
    int radius = 2;
    bool circular = true;           // eliminate tiles beyond radius + circular_fraction
    float circular_fraction = 0.5f; // fraction added to int radius to filter a circle

    std::function<float(float)> distance_formula = [](float distance) {
        return 1.0f / distance;
    };

    int quarter_mode = 2; // 0 = one quater, 1 = half, 2 = full

    void compute() {
        _starting_data();
        _normalize_weights();

        if (quarter_mode >= 1) _duplicate_as_rotation(1, false); // goes from a quater to half
        if (quarter_mode >= 2) _duplicate_as_rotation(2, true);  // half to full
    }

    void init() {
        radius = 2;
        circular = true;
        circular_fraction = 0.5f;
        quarter_mode = 2;
    }

    SamplingFieldGenerator() {
        init();
    }

    // just neighbours
    void PRESET_8_WAY() {
        init();
        radius = 1;
    }

    // diamond shape
    void PRESET_12_WAY() {
        init();
        radius = 2;
        circular_fraction = 0.0f;
    }

    // roughly circular
    void PRESET_20_WAY() {
        init();
        radius = 2;
        circular_fraction = 0.5f;
    }

    // roughly circular
    void PRESET_28_WAY() {
        init();
        radius = 3;
        circular_fraction = 0.0f;
    }

    // roughly circular
    void PRESET_36_WAY() {
        init();
        radius = 3;
        circular_fraction = 0.5f;
    }

    void _print_test_data() {
        compute();
        auto tiles = get_offsets();
        printf("%d\n", tiles.size());
        printf("%s\n", grid_positions_to_string(tiles).c_str());
        printf("--------------------------------\n");
    }

    void print_test_data() {

        PRESET_8_WAY();
        _print_test_data();

        PRESET_12_WAY();
        _print_test_data();

        PRESET_20_WAY();
        _print_test_data();

        PRESET_28_WAY();
        _print_test_data();

        PRESET_36_WAY();
        _print_test_data();
    }
};

#pragma endregion

#pragma endregion

} // namespace core::cuda::grid