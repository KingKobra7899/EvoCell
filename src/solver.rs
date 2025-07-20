use nalgebra::Vector2;
mod quadtree;
use quadtree::{QuadTree, Rect};
pub struct PhysicsSolver{
    positions: Vec<Vector2<f32>>,
    old_positions: Vec<Vector2<f32>>,
    accelerations: Vec<Vector2<f32>>,
    masses: Vec<f32>,
    radii: Vec<f32>,
    width: i32,
    height: i32,
    qt: QuadTree,
    center: Vector2<f32>,
    num_particles: i32
}

impl PhysicsSolver{
    pub fn new(width: i32, height: i32) -> PhysicsSolver{
        return PhysicsSolver { positions: Vec::new(), old_positions: Vec::new(), 
            accelerations: Vec::new(), masses: Vec::new(), radii: Vec::new(), 
            width: width, height: height, num_particles: 0, 
            center: Vector2::new((width as f32) / 2.0, (height as f32) / 2.0),
            qt: QuadTree::new(Rect::new((width as f32) / 2.0, (height as f32) / 2.0, (width as f32) / 2.0, (height as f32) / 2.0), 5)}
    }

    pub fn add_particle_grid(
        &mut self,
        num_x: usize,
        num_y: usize,
        start_pos: Vector2<f32>,
        radius: f32,
        spacing: f32,
        mass: f32,
    ) {
        for y in 0..num_y {
            for x in 0..num_x {
                let pos_x = start_pos.x + x as f32 * spacing;
                let pos_y = start_pos.y + y as f32 * spacing;
                self.add_particle(Vector2::new(pos_x, pos_y), mass, radius);
            }
        }
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

    pub fn update_quadtree(&mut self) {
        self.qt.clear();
        
        for i in 0..self.num_particles as usize {
            self.qt.insert(&self.positions[i], i as i32); // Adjust based on your QuadTree's insert signature
        }
    }

    pub fn inter_particle_collisions(&mut self){
        for i in 0..(self.num_particles as usize){
            let pos: Vector2<f32> = self.positions[i];
            let r: f32 = self.radii[i];
            let col_box = Rect::new(pos.x, pos.y, 3.0 * r, 3.0 * r);

            let indices = self.qt.query(&col_box);

            for n in indices{
                if n == (i as i32){
                    continue;
                }
                
                let other_pos: Vector2<f32> = self.positions[n as usize];
                let other_r: f32 = self.radii[n as usize];

                let col_axis: Vector2<f32> = pos - other_pos;
                let dist = col_axis.magnitude();

                if dist < (r + other_r){
                    let norm: Vector2<f32> = col_axis / dist;

                    let overlap: f32 =  (r + other_r) - dist;

                    let sep1: f32 = overlap * (other_r / (r + other_r));
                    let sep2: f32 = overlap * (r / (r + other_r));
                    
                    self.positions[i] += 0.5 * norm * sep1;
                    self.positions[n as usize] -= 0.5 * norm * sep2;
                }
            }
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

    pub fn update(&mut self, dt: f32, substeps: i32, grav: Vector2<f32>){
        self.update_quadtree();
        if substeps > 1{
            for _ in 0..substeps {
                self.integrate_forces(dt / (substeps as f32), grav);
                self.inter_particle_collisions();
                self.apply_circular_constraint(250.0);
            }
        }else{
            self.integrate_forces(dt, grav);
            self.inter_particle_collisions();
            self.apply_circular_constraint(250.0);
        }
    }
}