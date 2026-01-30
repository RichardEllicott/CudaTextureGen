/*

*/
shader_type spatial;

//uniform sampler2D albedo_tex : hint_default_white;

uniform vec4 albedo : source_color;
uniform sampler2D texture_albedo : source_color, filter_linear_mipmap, repeat_enable;

uniform float edge_width : hint_range(0.0, 0.2) = 0.02;

void fragment() {
    vec2 uv = UV;

    // Distance to nearest edge
    float dist_x = min(uv.x, 1.0 - uv.x);
    float dist_y = min(uv.y, 1.0 - uv.y);
    float dist = min(dist_x, dist_y);

    float fw = fwidth(dist);

    float edge_mask = 1.0 - smoothstep(edge_width - fw, edge_width + fw, dist);

    // --- Normal perturbation (bevel illusion) ---
    #define NORMAL_MODE 1
    #if NORMAL_MODE == 0
        vec3 n = NORMAL;
        n = normalize(n + edge_mask * vec3(0.0, 0.0, 1.0));
        NORMAL = n;
    #elif NORMAL_MODE == 1
        float b = pow(edge_mask, 0.5);
        vec3 n = NORMAL;
        n = normalize(n + b * vec3(0.0, 0.0, 1.0));
        NORMAL = n;
        
    #elif NORMAL_MODE == 2
    // Cubic falloff for a rounder bevel
    float b = edge_mask;
    b = b * b * (3.0 - 2.0 * b); // smoothstep curve

    // Build a tangent-space "bevel normal" pointing outward from the edge
    // This pushes the normal *sideways* instead of straight up.
    vec3 bevel_dir = normalize(vec3(
        (uv.x < 0.5 ? -1.0 : 1.0),
        (uv.y < 0.5 ? -1.0 : 1.0),
        0.0
    ));

    // Convert bevel_dir from tangent space to world space
    mat3 tbn = mat3(TANGENT, BINORMAL, NORMAL);
    vec3 bevel_normal = normalize(tbn * bevel_dir);

    // Blend between original normal and bevel normal
    NORMAL = normalize(mix(NORMAL, bevel_normal, b));

    #endif

    // --- Albedo texture ---
    
	vec4 albedo_tex = texture(texture_albedo, uv);
	ALBEDO = albedo.rgb * albedo_tex.rgb;
}
