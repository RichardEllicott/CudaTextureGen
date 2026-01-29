shader_type spatial;

// --- Normal map layers ---
uniform sampler2D normalA : hint_normal;
uniform sampler2D normalB : hint_normal;
uniform sampler2D normalC : hint_normal;
uniform sampler2D normalD : hint_normal;


uniform sampler2D gradient_tex : hint_default_white;
uniform float brightness = 1.0;


// per-layer UV scale (tiling)
uniform vec2 scaleA = vec2(1.0, 1.0);
uniform vec2 scaleB = vec2(1.0, 1.0);
uniform vec2 scaleC = vec2(1.0, 1.0);
uniform vec2 scaleD = vec2(1.0, 1.0);

// per-layer scrolling (UV units per second)
uniform vec2 scrollA = vec2(0.12, -0.11);
uniform vec2 scrollB = vec2(0.23, -0.13);
uniform vec2 scrollC = vec2(0.5, 0.01);
uniform vec2 scrollD = vec2(-0.02, 0.5);




#define ENABLE_WARP
#ifdef ENABLE_WARP
// uniform vec2 warpA = vec2(0, 0);
// uniform vec2 warpB = vec2(0, 0);
// uniform vec2 warpC = vec2(0, 0);

uniform float warpA = 0.0f;
uniform float warpB = 0.0f;


#endif

// per-layer speed multipliers (time)
uniform float speedA = 1.0;
uniform float speedB = 1.0;
uniform float speedC = 1.0;
uniform float speedD = 1.0;

#define MAX_STRENGTH 32
#define DEFAULT_STRENGTH 0

// per-layer strength (scale XY of decoded normal to control detail)
uniform float strengthA : hint_range(0.0, MAX_STRENGTH) = DEFAULT_STRENGTH;
uniform float strengthB : hint_range(0.0, MAX_STRENGTH) = DEFAULT_STRENGTH;
uniform float strengthC : hint_range(0.0, MAX_STRENGTH) = DEFAULT_STRENGTH;
uniform float strengthD : hint_range(0.0, MAX_STRENGTH) = DEFAULT_STRENGTH;

// overall blend factor (0..1) to mix geometric normal with combined detail
uniform float detail_blend = 1.0;

uniform float speed : hint_range(0.0, 16.0) = 1.0;

// small epsilon to avoid degenerate tangents
const float EPS = 1e-6;

// Decode normal from texture (0..1 -> -1..1)
vec3 decode_normal(vec3 c) {
    return c * 2.0 - 1.0;
}

// Reoriented Normal Mapping (RNM) combine two tangent-space normals a and b
vec3 rnm_combine(vec3 a, vec3 b) {
    vec2 ax = a.xy;
    vec2 bx = b.xy;
    float az = a.z;
    float bz = b.z;
    vec2 xy = ax * bz + bx * az;
    float z = az * bz - dot(ax, bx);
    return normalize(vec3(xy, z));
}

void fragment() {
    // base UV
    //vec2 uv = UV;


    vec3 world_pos = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
    vec2 uv = world_pos.xz / 64.0;
        
    
    
    vec3 color = vec3(1,1,1);
    
    vec2 uvA = uv * scaleA;
    vec2 uvB = uv * scaleB;
    vec2 uvC = uv * scaleC;
    vec2 uvD = uv * scaleD;
    


    uvA += scrollA * TIME * speedA * speed;
    uvB += scrollB * TIME * speedB * speed;
    uvC += scrollC * TIME * speedC * speed;
    uvD += scrollD * TIME * speedD * speed;
    
    // use nd to offset the others

    vec3 _nd = texture(normalD, uvD).rgb;
    vec3 _nc = texture(normalC, uvC).rgb;

    #ifdef ENABLE_WARP
    uvA += vec2(_nd.g, _nc.g) * warpA;
    uvB += vec2(_nd.g, _nc.g) * warpB;
    //uvC += warpC * _nd.b;
    #endif
    
    
    vec3 _na = texture(normalA, uvA).rgb;
    vec3 _nb = texture(normalB, uvB).rgb;
    
    
    vec3 grad_color = texture(gradient_tex, vec2(_na.r, 0.5)).rgb;
    color *= grad_color;
    color *= brightness;

    // sample and decode normals
    vec3 na = decode_normal(_na);
    vec3 nb = decode_normal(_nb);
    vec3 nc = decode_normal(_nc);
    vec3 nd = decode_normal(_nd);
    
    //color = vec3(_nd.g, _nd.g, _nd.g)

    // apply per-layer strength by scaling XY (preserves Z via renormalize)
    na.xy *= strengthA;
    nb.xy *= strengthB;
    nc.xy *= strengthC;
    nd.xy *= strengthD;
    na = normalize(na);
    nb = normalize(nb);
    nc = normalize(nc);
    nd = normalize(nd);
    
    

    // combine with RNM iteratively: ((A+B)+C)+D
    vec3 acc = rnm_combine(na, nb);
    acc = rnm_combine(acc, nc);
    acc = rnm_combine(acc, nd);

    // Build TBN in model/local space (robust to TANGENT being vec3)
    vec3 t = TANGENT;
    if (length(t) < EPS) {
        t = vec3(1.0, 0.0, 0.0);
    }
    t = normalize(t);
    vec3 n = normalize(NORMAL);

    // compute bitangent and ensure correct handedness without relying on tangent.w
    vec3 b = normalize(cross(n, t));
    if (dot(cross(t, b), n) < 0.0) {
        b = -b;
    }

    mat3 tbn = mat3(t, b, n);

    // Convert combined tangent-space normal to model/local space
    vec3 model_space_normal = normalize(tbn * acc);

    // Convert model-space normal to world-space direction (w = 0 so translation ignored)
    vec3 world_space_normal = normalize((MODEL_MATRIX * vec4(model_space_normal, 0.0)).xyz);

    // Geometric normal in world space
    vec3 geom_world_normal = normalize((MODEL_MATRIX * vec4(n, 0.0)).xyz);

    // Final normal: blend between geometry and detail
    vec3 final_world_normal = normalize(mix(geom_world_normal, world_space_normal, clamp(detail_blend, 0.0, 1.0)));

    // Assign final normal for lighting
    NORMAL = final_world_normal;
    
    ALBEDO = color;

    // (Optional) you can still set ALBEDO, ROUGHNESS, etc. here
}
