shader_type canvas_item;

uniform sampler2D SCREEN_TEXTURE : hint_screen_texture, filter_linear_mipmap;
uniform sampler2D noise_texture;

uniform float chromatic_aberration = 4.32;
uniform float lens_distortion = 1.12;
uniform float line_amount = 200.0;
uniform float lines_speed = 1.0;
uniform float lines_displacement = 0.64;
uniform float sine_lines_amount = 14.9;
uniform float sine_lines_speed = 20.0;
uniform float sine_lines_displacement = 10.5;
uniform float sine_lines_threshold = 0.055;
uniform float contrast = 0.924;
uniform float noise_strength = 0.15;

void fragment() {
    vec2 uv = SCREEN_UV;

    // Lens distortion
    vec2 center = vec2(0.5);
    vec2 to_center = uv - center;
    float dist = length(to_center);
    uv += to_center * dist * lens_distortion * 0.01;

    // Sine wave scanline distortion
    float sine_wave = sin(uv.y * sine_lines_amount + TIME * sine_lines_speed);
    if (abs(sine_wave) < sine_lines_threshold) {
        uv.x += sine_wave * sine_lines_displacement * 0.001;
    }

    // Horizontal flicker lines
    float line_offset = sin(uv.y * line_amount + TIME * lines_speed) * lines_displacement * 0.001;
    uv.x += line_offset;

    // Chromatic aberration
    vec4 col;
    col.r = texture(SCREEN_TEXTURE, uv + vec2(chromatic_aberration * 0.001, 0.0)).r;
    col.g = texture(SCREEN_TEXTURE, uv).g;
    col.b = texture(SCREEN_TEXTURE, uv - vec2(chromatic_aberration * 0.001, 0.0)).b;
    col.a = 1.0;

    // Add noise flicker
    vec2 noise_uv = vec2(uv.x, uv.y * 20.0 + TIME * 10.0); // scroll vertically
    float noise = texture(noise_texture, noise_uv).r;
    col.rgb += noise * noise_strength;

    // Contrast adjustment
    col.rgb = (col.rgb - 0.5) * contrast + 0.5;

    COLOR = col;
}
