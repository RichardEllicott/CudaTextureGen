/*
fake bevel like effect based on uv pos

*/
shader_type spatial;

//uniform sampler2D albedo_tex : hint_default_white;

uniform vec4 albedo : source_color = vec4(1,1,1,1);
uniform sampler2D texture_albedo: hint_default_white;

uniform float edge_width : hint_range(0.0, 1.0) = 0.02;

uniform float normal_map_depth : hint_range(0.0, 16.0) = 1.0;

uniform bool cubic_ease = false;

uniform bool apply_edge_mask_to_albedo = false; // debug


uniform float ao_strength : hint_range(0.0, 16.0)  = 0.0;
uniform float ao_light_affect : hint_range(0.0, 16.0)  = 0.0;

//uniform float amb

float _cubic_ease(float t){
    return t * t * (3.0 - 2.0 * t); // cubic ease
}


void fragment() {
    vec2 uv = UV;

    vec2 uv_mod = fract(UV); // Wrap UVs so edges always fall in 0–1

    // Distance to left/right edges only
    float dist_x = min(uv_mod.x, 1.0 - uv_mod.x);
    float dist_y = min(uv_mod.y, 1.0 - uv_mod.y);

    // Smooth fade
    float edge_mask_x = smoothstep(0.0, edge_width, dist_x);
    float edge_mask_y = smoothstep(0.0, edge_width, dist_y);

    if (cubic_ease){
        edge_mask_x = _cubic_ease(edge_mask_x);
        edge_mask_y = _cubic_ease(edge_mask_y);
    }
    
    float edge_mask = edge_mask_x * edge_mask_y;
    
    //vec3 n = NORMAL;
    vec3 n = NORMAL_MAP;
    
    NORMAL_MAP_DEPTH = normal_map_depth;
    
    n.x *= edge_mask_x;
    n.y *= edge_mask_y;
    NORMAL_MAP = normalize(n);

    // Sample texture
    vec4 _albedo = texture(texture_albedo, uv);
    _albedo *= albedo;
    

    if (apply_edge_mask_to_albedo){_albedo *= edge_mask;}
    
    
    AO = mix(1.0, edge_mask, ao_strength);
    
    AO_LIGHT_AFFECT = ao_light_affect;

    ALBEDO = _albedo.rgb;
}

