use nalgebra::Vector2;
use raqote::{DrawTarget, DrawOptions, PathBuilder, SolidSource, Source};

pub struct PhysicsSolver{
    positions: Vec<Vector2<f32>>,
    old_positions: Vec<Vector2<f32>>,
    accelerations: Vec<Vector2<f32>>,
    masses: Vec<f32>,
    radii: Vec<f32>,
    width: i32,
    height: i32,
    center: Vector2<f32>,
    num_particles: i32
}

impl PhysicsSolver{
    pub fn new(width: i32, height: i32) -> PhysicsSolver{
        return PhysicsSolver { positions: Vec::new(), old_positions: Vec::new(), accelerations: Vec::new(), masses: Vec::new(), radii: Vec::new(), width: width, height: height, num_particles: 0, center: Vector2::new((width as f32) / 2.0, (height as f32) / 2.0)}
    }

    pub fn add_particle(&mut self, pos: Vector2<f32>, mass: f32, radius: f32){
        self.positions.push(pos);
        self.old_positions.push(pos);
        self.accelerations.push(Vector2::new(0.0,0.0));
        self.masses.push(mass);
        self.radii.push(radius);
        self.num_particles = self.num_particles + 1;
    }
    
    pub fn accelerate_particle(&mut self, index: usize, force: Vector2<f32>){
        self.accelerations[index] += force / self.masses[index];
    }

    pub fn integrate_forces(&mut self, dt: f32, grav: Vector2<f32>){
        for i in 0..(self.num_particles as usize){
            self.accelerate_particle(i, grav * self.masses[i]);
            
            let pos: Vector2<f32> = self.positions[i];
            let old_pos: Vector2<f32> = self.old_positions[i];
            let acc: Vector2<f32> = self.accelerations[i];

            let new_pos: Vector2<f32> = (pos) + (pos - old_pos) + acc * dt * dt;
            self.old_positions[i] = pos;
            self.positions[i] = new_pos;
            self.accelerations[i] = Vector2::new(0.0,0.0);
        }
    }

    pub fn render(&self, dt: &mut DrawTarget) {
        
        let mut min_vel_mag: f32 = f32::INFINITY;
        let mut max_vel_mag: f32 = 0.0;
        
        for i in 0..(self.num_particles as usize) {
            let vel: Vector2<f32> = self.positions[i] - self.old_positions[i]; 
            let vel_magnitude: f32 = vel.magnitude();
            min_vel_mag = min_vel_mag.min(vel_magnitude);
            max_vel_mag = max_vel_mag.max(vel_magnitude);
        }
        
        
        let vel_range = if max_vel_mag > min_vel_mag { max_vel_mag - min_vel_mag } else { 1.0 };
        
        for i in 0..(self.num_particles as usize) {
            let pos = self.positions[i];
            let radius = self.radii[i];
            let vel: Vector2<f32> = self.positions[i] - self.old_positions[i];
            
            if !pos.x.is_finite() || !pos.y.is_finite() || !radius.is_finite() {
                eprintln!("Invalid values at particle {}: pos=({}, {}), radius={}", 
                         i, pos.x, pos.y, radius);
                continue;
            }

            let vel_magnitude = (vel.x * vel.x + vel.y * vel.y).sqrt();
            
            
            let normalized_vel = (vel_magnitude - min_vel_mag) / vel_range;
            
            
            let red_value = (normalized_vel * 255.0) as u8;
            
            let mut path = PathBuilder::new();
            path.arc(pos.x, pos.y, radius, 0.0, 2.0 * std::f32::consts::PI);
            let circle_path = path.finish();
    
            dt.fill(
                &circle_path,
                &Source::Solid(SolidSource::from_unpremultiplied_argb(
                    255,        
                    red_value,  
                    100,        
                    150         
                )),
                &DrawOptions::new(),
            );
        }
    }

    pub fn apply_circular_constraint(&mut self, con_radius: f32){
        for i in 0..(self.num_particles as usize){
            let mut pos: Vector2<f32> = self.positions[i];
            let radius: f32 = self.radii[i];

            let to_obj: Vector2<f32> = pos - self.center;
            let dist: f32 = to_obj.magnitude();

            if dist > (con_radius - radius){
                let n: Vector2<f32> = to_obj / dist;
                let new_pos: Vector2<f32> = self.center + n * (con_radius - radius);
                self.positions[i] = new_pos;
            }
        }
    }

    pub fn update(&mut self, draw_target: &mut DrawTarget, dt: f32, substeps: i32, grav: Vector2<f32>){
        if substeps > 1{
            for _ in 0..substeps {
                self.integrate_forces(dt / (substeps as f32), grav);
                self.apply_circular_constraint(250.0);
            }
        }else{
            self.integrate_forces(dt, grav);
            self.apply_circular_constraint(250.0);
        }
        self.render(draw_target);
    }
}