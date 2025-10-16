/*

custom types trying to match Godot's patterns (might not be used due to preferance for flat float arrays)

*/

#pragma once

#include <cmath>

struct Vector2i {
    int x = 0;
    int y = 0;

    Vector2i() = default;
    Vector2i(int x, int y) : x(x), y(y) {}

    Vector2i operator+(const Vector2i &other) const { return {x + other.x, y + other.y}; }
    Vector2i operator-(const Vector2i &other) const { return {x - other.x, y - other.y}; }
    bool operator==(const Vector2i &other) const { return x == other.x && y == other.y; }
};

struct Vector3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vector3() = default;
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vector3 operator+(const Vector3 &other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vector3 operator-(const Vector3 &other) const { return {x - other.x, y - other.y, z - other.z}; }

    Vector3 normalized() const {
        float len = std::sqrt(x * x + y * y + z * z);
        return len > 0 ? Vector3{x / len, y / len, z / len} : Vector3{0, 0, 0};
    }
};

struct Color {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;

    Color() = default;
    Color(float r, float g, float b, float a = 1.0f) : r(r), g(g), b(b), a(a) {}
};
