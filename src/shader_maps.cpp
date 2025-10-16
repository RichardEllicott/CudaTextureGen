#include "shader_maps.h"

namespace shader_maps {


/*
int posmod(int value, int mod) {
    int result = value % mod;
    return result < 0 ? result + mod : result;
}

int image_position_to_index(const Vector2i &pos, const Vector2i &size, bool wrap) {
    int x = pos.x, y = pos.y;
    if (wrap) {
        x = posmod(x, size.x);
        y = posmod(y, size.y);
    } else {
        x = std::clamp(x, 0, size.x - 1);
        y = std::clamp(y, 0, size.y - 1);
    }
    return y * size.x + x;
}

Vector3 normalize(const Vector3 &v) {
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return len > 0.0f ? Vector3{v.x / len, v.y / len, v.z / len} : Vector3{0, 0, 0};
}

std::vector<Vector3> generate_normal_map_array(const std::vector<float> &image, Vector2i image_size, float normal_scale, bool wrap) {
    std::vector<Vector3> output(image_size.x * image_size.y);

    auto get_pixel = [&](Vector2i pos) -> float {
        if (wrap) {
            pos.x = (pos.x + image_size.x) % image_size.x;
            pos.y = (pos.y + image_size.y) % image_size.y;
        } else {
            pos.x = std::clamp(pos.x, 0, image_size.x - 1);
            pos.y = std::clamp(pos.y, 0, image_size.y - 1);
        }
        return image[pos.y * image_size.x + pos.x];
    };

    for (int y = 0; y < image_size.y; ++y) {
        for (int x = 0; x < image_size.x; ++x) {
            Vector2i pos(x, y);

            float tl = get_pixel(pos + Vector2i(-1, -1));
            float t = get_pixel(pos + Vector2i(0, -1));
            float tr = get_pixel(pos + Vector2i(1, -1));
            float l = get_pixel(pos + Vector2i(-1, 0));
            float r = get_pixel(pos + Vector2i(1, 0));
            float bl = get_pixel(pos + Vector2i(-1, 1));
            float b = get_pixel(pos + Vector2i(0, 1));
            float br = get_pixel(pos + Vector2i(1, 1));

            float dx = (tr + 2 * r + br) - (tl + 2 * l + bl);
            float dy = (bl + 2 * b + br) - (tl + 2 * t + tr);

            Vector3 normal(-dx * normal_scale, -dy * normal_scale, 1.0f);
            normal = normal.normalized();

            output[y * image_size.x + x] = normal;
        }
    }

    return output;
}

std::vector<float> generate_ao_map_array(const std::vector<float> &image, Vector2i image_size, float radius = 5.0f, float strength = 1.0f, bool wrap = false) {
    std::vector<float> output(image_size.x * image_size.y);

    auto get_pixel = [&](Vector2i pos) -> float {
        if (wrap) {
            pos.x = (pos.x + image_size.x) % image_size.x;
            pos.y = (pos.y + image_size.y) % image_size.y;
        } else {
            pos.x = std::clamp(pos.x, 0, image_size.x - 1);
            pos.y = std::clamp(pos.y, 0, image_size.y - 1);
        }
        return image[pos.y * image_size.x + pos.x];
    };

    for (int y = 0; y < image_size.y; ++y) {
        for (int x = 0; x < image_size.x; ++x) {
            Vector2i pos(x, y);
            float center_height = get_pixel(pos);

            float occlusion = 0.0f;
            int sample_count = 0;

            for (int dy = -static_cast<int>(radius); dy <= static_cast<int>(radius); ++dy) {
                for (int dx = -static_cast<int>(radius); dx <= static_cast<int>(radius); ++dx) {
                    if (dx == 0 && dy == 0)
                        continue;

                    Vector2i sample_pos = pos + Vector2i(dx, dy);
                    float sample_height = get_pixel(sample_pos);

                    float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy));
                    float height_diff = center_height - sample_height;

                    if (distance > 0.0f) {
                        occlusion += std::max(height_diff / distance, 0.0f);
                        ++sample_count;
                    }
                }
            }

            float ao = 1.0f - std::clamp(occlusion * strength / sample_count, 0.0f, 1.0f);
            output[y * image_size.x + x] = ao;
        }
    }

    return output;
}






*/

} // namespace shader_maps
