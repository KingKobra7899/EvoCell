
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

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let pixel_coord = in.frag_coord.xy;
    var final_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    
    let EDGE_SMOOTHNESS = 1.0; 
    let PARTICLE_BASE_COLOR = vec4<f32>(0.2, 0.6, 1.0, 0.8); 
    

    for (var i: u32 = 0u; i < num_particles_uniform; i = i + 1u) {
        let p = particles[i];
        let distance = length(pixel_coord - p.position);

        if (distance < p.radius) {
            let alpha_falloff = smoothstep(1.0, 1.0 - EDGE_SMOOTHNESS, distance / p.radius);
            let particle_color_with_falloff = vec4<f32>(PARTICLE_BASE_COLOR.rgb, PARTICLE_BASE_COLOR.a * alpha_falloff);
            final_color = final_color * (1.0 - particle_color_with_falloff.a) + particle_color_with_falloff;
        }
    }
    return final_color;
}