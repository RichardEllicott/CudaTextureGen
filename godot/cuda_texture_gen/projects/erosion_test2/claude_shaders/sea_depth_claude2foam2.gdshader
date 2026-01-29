shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, diffuse_burley, specular_schlick_ggx;

// Depth texture (required for Godot 4.x)
uniform sampler2D DEPTH_TEXTURE : hint_depth_texture, filter_linear_mipmap;
uniform sampler2D SCREEN_TEXTURE : hint_screen_texture, filter_linear_mipmap;

// Water colors
uniform vec4 foam_color : source_color = vec4(1.0, 1.0, 1.0, 0.9);
uniform vec4 shallow_color : source_color = vec4(0.0, 0.8, 0.9, 0.8);
uniform vec4 deep_color : source_color = vec4(0.0, 0.2, 0.5, 1.0);

// Depth distances
uniform float foam_distance : hint_range(0.0, 5.0) = 0.5;
uniform float shallow_distance : hint_range(0.0, 50.0) = 10.0;

// Wave properties
uniform float wave_speed : hint_range(0.0, 2.0) = 0.5;
uniform float wave_scale : hint_range(0.0, 1.0) = 0.2;
uniform float wave_length : hint_range(0.1, 20.0) = 2.0;
uniform vec2 wave_direction = vec2(1.0, 0.0);

// Rendering
uniform float metallic : hint_range(0.0, 1.0) = 0.0;
uniform float roughness : hint_range(0.0, 1.0) = 0.02;

void vertex() {
    // Simple wave animation with adjustable wavelength
    
        //vec3 WORLD_VERTEX = (INV_VIEW_MATRIX  * vec4(VERTEX, 1.0)).xyz;
        vec3 WORLD_VERTEX = (MODEL_MATRIX  * vec4(VERTEX, 1.0)).xyz;
    //vec3 WORLD_VERTEX = VERTEX;
    
    float wave_coord = (WORLD_VERTEX.x * wave_direction.x + WORLD_VERTEX.z * wave_direction.y) / wave_length;
    float wave = sin(TIME * wave_speed + wave_coord) * wave_scale;
    VERTEX.y += wave;
}

float get_depth_at_screen_uv(vec2 screen_uv, mat4 inv_proj) {
    float raw_depth = texture(DEPTH_TEXTURE, screen_uv).r;
    vec3 ndc = vec3(screen_uv * 2.0 - 1.0, raw_depth);
    vec4 view = inv_proj * vec4(ndc, 1.0);
    view.xyz /= view.w;
    return -view.z;
}

void fragment() {
    // Get the depth of the terrain/objects behind the water
    float background_depth = get_depth_at_screen_uv(SCREEN_UV, INV_PROJECTION_MATRIX);
    
    // Get the depth of the water surface itself
    float water_depth = FRAGCOORD.z;
    vec3 ndc = vec3(SCREEN_UV * 2.0 - 1.0, water_depth);
    vec4 view = INV_PROJECTION_MATRIX * vec4(ndc, 1.0);
    view.xyz /= view.w;
    float water_surface_depth = -view.z;
    
    // Calculate the depth difference
    float depth_difference = background_depth - water_surface_depth;
    
    // Calculate depth fades
    float foam_fade = clamp(depth_difference / foam_distance, 0.0, 1.0);
    float shallow_fade = clamp((depth_difference - foam_distance) / shallow_distance, 0.0, 1.0);
    
    // Mix colors: foam -> shallow -> deep
    vec4 water_color = mix(foam_color, shallow_color, foam_fade);
    water_color = mix(water_color, deep_color, shallow_fade);
    
    ALBEDO = water_color.rgb;
    ALPHA = water_color.a;
    METALLIC = metallic;
    ROUGHNESS = roughness;
}