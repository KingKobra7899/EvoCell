
struct Particle {
    position: vec2<f32>,
    radius: f32,
    _padding: f32,
};

@group(0) @binding(0)
var<storage, read> particles: array<Particle>;

@group(0) @binding(1)
var<uniform> num_particles_uniform: u32;

@group(1) @binding(0)
var<uniform> screen_dims: vec2<f32>;

struct FragmentInput {
    @builtin(position) frag_coord: vec4<f32>,
};

fn gamma_correct(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / 2.2)); // Approximate sRGB gamma
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let pixel_coord = in.frag_coord.xy;

    // Background color: dark blueish
    var final_color = vec4<f32>(0,0,0, 1.0);

    for (var i: u32 = 0u; i < num_particles_uniform; i = i + 1u) {
        let p = particles[i];
        let dist = length(pixel_coord - p.position);

        if (dist < p.radius) {
            let norm = dist / p.radius;

            let membrane_thickness = 0.25;
            let membrane = step(1.0 - membrane_thickness, norm) - step(1.0, norm);
            let interior = step(0.0, norm) - step(1.0 - membrane_thickness, norm);
            let interior_brightness = 0.1;

            let membrane_color = vec3<f32>(0.2, 0.5, 1.0); // bright membrane
            let interior_color = vec3<f32>(0.1, 0.05, 0.05); // dim red

            let color = membrane * membrane_color + interior * interior_color;
            let alpha = membrane + interior * interior_brightness;

            // Safely reassign with updated color
            final_color = vec4<f32>(final_color.rgb + color * alpha, 1.0);
        }
    }

    return vec4<f32>(gamma_correct(final_color.rgb), 1.0);

}