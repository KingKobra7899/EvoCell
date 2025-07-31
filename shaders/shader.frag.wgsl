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
                // PLANT CELL - Green with chloroplast-like structures
                let membrane_thickness = 0.15;
                let cell_wall_thickness = 0.05;
                
                // Cell wall (outermost layer)
                let cell_wall = step(1.0 - cell_wall_thickness, norm) - step(1.0, norm);
                
                // Cell membrane
                let membrane_start = 1.0 - cell_wall_thickness - membrane_thickness;
                let membrane = step(membrane_start, norm) - step(1.0 - cell_wall_thickness, norm);
                
                // Interior cytoplasm
                let interior = step(0.0, norm) - step(membrane_start, norm);
                
                // Add chloroplast-like structures using noise
                let chloroplast_noise = noise((pixel_coord - p.position) * 0.1);
                let chloroplast_pattern = step(0.6, chloroplast_noise) * interior;
                
                // Plant colors
                let cell_wall_color = vec3<f32>(0.4, 0.7, 0.3);     // Light green wall
                let membrane_color = vec3<f32>(0.2, 0.8, 0.4);      // Bright green membrane  
                let cytoplasm_color = vec3<f32>(0.1, 0.3, 0.1);     // Dark green cytoplasm
                let chloroplast_color = vec3<f32>(0.0, 0.6, 0.2);   // Chloroplast green
                
                let color = cell_wall * cell_wall_color +
                           membrane * membrane_color +
                           interior * cytoplasm_color +
                           chloroplast_pattern * chloroplast_color;
                
                let alpha = cell_wall + membrane + interior * 0.8 + chloroplast_pattern * 0.5;
                final_color += color * alpha;
                
            } else {
                // ANIMAL CELL - Pink/red with organelle-like structures
                let membrane_thickness = 0.2;
                let membrane = step(1.0 - membrane_thickness, norm) - step(1.0, norm);
                let interior = step(0.0, norm) - step(1.0 - membrane_thickness, norm);
                
                // Add nucleus and organelles using noise
                let nucleus_center = length((pixel_coord - p.position) / p.radius);
                let nucleus = step(nucleus_center, 0.3) * interior;
                
                let organelle_noise = noise((pixel_coord - p.position) * 0.15);
                let organelle_pattern = step(0.7, organelle_noise) * interior * (1.0 - nucleus);
                
                // Animal cell colors
                let membrane_color = vec3<f32>(0.8, 0.3, 0.5);      // Pink membrane
                let cytoplasm_color = vec3<f32>(0.4, 0.1, 0.2);     // Dark red cytoplasm
                let nucleus_color = vec3<f32>(0.3, 0.1, 0.4);       // Purple nucleus
                let organelle_color = vec3<f32>(0.6, 0.2, 0.3);     // Organelle color
                
                let color = membrane * membrane_color +
                           interior * cytoplasm_color +
                           nucleus * nucleus_color +
                           organelle_pattern * organelle_color;
                
                let alpha = membrane + interior * 0.6 + nucleus * 0.8 + organelle_pattern * 0.4;
                final_color += color * alpha;
            }
        }
    }
    
    return vec4<f32>(gamma_correct(final_color), 1.0);
}