shader_type spatial;

uniform sampler3D noise_3d;


// my test

//void fragment() {
    //vec3 WORLD_POS = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
    //vec2 WORLD_UV = WORLD_POS.xz;
    //
    //
    //float t = TIME / 64.0;
    //// Super fast lookup instead of calculation
    //float sample1 = texture(noise_3d, vec3(WORLD_UV, t)).r;
    //
    //float sample2 = texture(noise_3d, vec3(WORLD_UV / 3.5634, t + 3.2681)).r;
    //
    //float sample3 = texture(noise_3d, vec3(WORLD_UV / 7.325, t + 3.2681)).r;
//
    //sample1 *= sample2;    
    //ALBEDO = vec3(sample1);
//}

// fbm


uniform float speed : hint_range(0.0, 2.0) = 0.25;
uniform float scale : hint_range(0.1, 20.0) = 5.0;
//uniform int octaves : hint_range(1, 8) = 4;
//uniform float lacunarity : hint_range(1.0, 4.0) = 2.0;
//uniform float gain : hint_range(0.0, 1.0) = 0.5;
//
//// FBM function using 3D texture sampling
//float fbm(vec3 p) {
    //float value = 0.0;
    //float amplitude = 1.0;
    //float frequency = 1.0;
    //float max_value = 0.0;
    //
    //for (int i = 0; i < octaves; i++) {
        //// Sample the 3D texture at increasing frequencies
        //value += texture(noise_3d, p * frequency).r * amplitude;
        //
        //max_value += amplitude;
        //amplitude *= gain;
        //frequency *= lacunarity;
    //}
    //
    //// Normalize to [0, 1] range
    //return value / max_value;
//}
//
//void fragment() {
    //float t = TIME * speed / 64.0;
    //vec3 p = vec3(UV * scale, t);
    //
    //float n = fbm(p);
    //ALBEDO = vec3(n);
//}

// simpler faster
void fragment() {
    
    vec3 WORLD_POS = (INV_VIEW_MATRIX * vec4(VERTEX, 1.0)).xyz;
    vec2 WORLD_UV = WORLD_POS.xz;
    
    float t = TIME * speed;
    vec3 p = vec3(WORLD_UV * scale, t);
    
    // Just 2-3 octaves, no domain warp
    float n1 = texture(noise_3d, p * 1.0).r;
    float n2 = texture(noise_3d, p * 2.0).r * 0.5;
    float n3 = texture(noise_3d, p * 4.0).r * 0.25;
    
    float n = (n1 + n2 + n3) / 1.75; // Normalize
    ALBEDO = vec3(n);
}


// HYBRID??
// Strategy 2: Texture for domain warping, procedural for final sample
// https://claude.ai/chat/70e02ba2-6d12-4a79-ac82-1daa73c0cd5e

// non repatition!!

//uniform sampler3D warp_texture; // Pre-rendered noise for warping
//
//void fragment() {
    //float t = TIME * speed;
    //vec3 p = vec3(UV * scale, t);
    //
    //// Use texture to warp the domain (cheap 3D lookup)
    //vec3 warp = vec3(
        //texture(warp_texture, p + vec3(0.0, 0.0, 0.0)).r,
        //texture(warp_texture, p + vec3(5.2, 1.3, 0.0)).r,
        //texture(warp_texture, p + vec3(9.1, 2.7, 0.0)).r
    //);
    //
    //// Then calculate ONE procedural sample with warped coords
    //float n = simplex3(p + warp * 0.5);
    //
    //ALBEDO = vec3(n * 0.5 + 0.5);
//}




