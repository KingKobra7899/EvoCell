struct Particle {
    position: vec2<f32>,
    radius: f32,
    is_plant: u32
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
    return pow(color, vec3<f32>(1.0 / 2.2));
}

// Clean falloff for crisp, publication-ready circles
fn circle_falloff(dist: f32, radius: f32) -> f32 {
    let norm_dist = dist / radius;
    return 1.0 - smoothstep(0.9, 1.0, norm_dist);
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let pixel_coord = in.frag_coord.xy;
    
    var final_color = vec3<f32>(1.0, 1.0, 1.0); // White background for publication
    
    for (var i: u32 = 0u; i < num_particles_uniform; i = i + 1u) {
        let p = particles[i];
        let dist = length(pixel_coord - p.position);
        
        if (dist < p.radius) {
            let falloff = circle_falloff(dist, p.radius);
            
            if (p.is_plant == 1u) {
                // Producer organisms - clean dark green
                let base_color = vec3<f32>(0.2, 0.6, 0.2);
                final_color = mix(final_color, base_color, falloff);
            } else {
                // Consumer organisms - clean red
                let base_color = vec3<f32>(1.0, 0.2, 0.2);
                final_color = mix(final_color, base_color, falloff);
            }
        }
    }
    
    return vec4<f32>(final_color, 1.0);
}