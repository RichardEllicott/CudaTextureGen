shader_type spatial;

// ------------------------------------------------------------
// Dave Hoskins "Hash without Sine"
// ------------------------------------------------------------
vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

// ------------------------------------------------------------
// Proper simplex gradient set (12 directions)
// ------------------------------------------------------------
const vec3 grad3lut[12] = vec3[](
    vec3( 1, 1, 0), vec3(-1, 1, 0), vec3( 1,-1, 0), vec3(-1,-1, 0),
    vec3( 1, 0, 1), vec3(-1, 0, 1), vec3( 1, 0,-1), vec3(-1, 0,-1),
    vec3( 0, 1, 1), vec3( 0,-1, 1), vec3( 0, 1,-1), vec3( 0,-1,-1)
);

vec3 grad3(vec3 p) {
    vec3 h = hash33(p);
    int idx = int(mod(floor(h.x * 12.0), 12.0));
    return grad3lut[idx];
}

// ------------------------------------------------------------
// 3D Simplex Noise
// ------------------------------------------------------------
float simplex3(vec3 p) {
    const float F3 = 1.0 / 3.0;
    const float G3 = 1.0 / 6.0;
    
    float s = (p.x + p.y + p.z) * F3;
    vec3 i = floor(p + s);
    float t = (i.x + i.y + i.z) * G3;
    vec3 X0 = i - t;
    vec3 x0 = p - X0;
    
    // Determine simplex traversal order
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    
    vec3 x1 = x0 - i1 + G3;
    vec3 x2 = x0 - i2 + 2.0 * G3;
    vec3 x3 = x0 - 1.0 + 3.0 * G3;
    
    // Contribution from each corner
    float n = 0.0;
    
    float t0 = 0.6 - dot(x0, x0);
    if (t0 > 0.0) {
        t0 *= t0;
        n += t0 * t0 * dot(grad3(i), x0);
    }
    
    float t1 = 0.6 - dot(x1, x1);
    if (t1 > 0.0) {
        t1 *= t1;
        n += t1 * t1 * dot(grad3(i + i1), x1);
    }
    
    float t2 = 0.6 - dot(x2, x2);
    if (t2 > 0.0) {
        t2 *= t2;
        n += t2 * t2 * dot(grad3(i + i2), x2);
    }
    
    float t3 = 0.6 - dot(x3, x3);
    if (t3 > 0.0) {
        t3 *= t3;
        n += t3 * t3 * dot(grad3(i + 1.0), x3);
    }
    
    return 32.0 * n;
}

// ------------------------------------------------------------
// Shader parameters
// ------------------------------------------------------------
uniform float speed : hint_range(0.0, 2.0) = 0.25;
uniform float scale : hint_range(0.1, 20.0) = 5.0;

void fragment() {
    
    vec3 WORLD_POS = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
    vec2 WORLD_UV = WORLD_POS.xz;
    
    
    float t = TIME * speed;
    float n = simplex3(vec3(WORLD_UV * scale, t));
    
    // Remap from [-1, 1] to [0, 1]
    n = n * 0.5 + 0.5;
    
    ALBEDO = vec3(n);
}