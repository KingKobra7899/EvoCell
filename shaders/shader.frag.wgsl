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

// Generate noise for organic texture
fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(hash(i + vec2<f32>(0.0, 0.0)), hash(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let pixel_coord = in.frag_coord.xy;
    
    var final_color = vec3<f32>(0.0, 0.0, 0.0);
    
    for (var i: u32 = 0u; i < num_particles_uniform; i = i + 1u) {
        let p = particles[i];
        let dist = length(pixel_coord - p.position);
        
        if (dist < p.radius) {
            let norm = dist / p.radius;
            
            if (p.is_plant == 1u) {
                // PLANT CELL - Simple green circle
                let color = vec3<f32>(0.2, 0.8, 0.3);
                final_color += color;
                
            } else {
                // ANIMAL CELL - Simple blue circle
                let color = vec3<f32>(0.3, 0.5, 0.9);
                final_color += color;
            }
        }
    }
    
    return vec4<f32>(gamma_correct(final_color), 1.0);
}