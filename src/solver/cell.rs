use std::f32::consts::PI;

use nalgebra::{clamp, DMatrix, DVector, Vector2, VectorN};
use rand::{random_range, rngs::ThreadRng, seq::SliceRandom as _, Rng};
use rand_distr::{Normal, Distribution};

use crate::solver::PhysicsSolver;

const MUTATION_RATE:f64 = 0.1;
pub struct EnvironmentalEncoder{
    pub(crate) weight_matrix: DMatrix<f32>,
    pub(crate) bias: DMatrix<f32>,
}

impl EnvironmentalEncoder {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weight_matrix = DMatrix::<f32>::zeros(output_size, input_size);
        let bias = DMatrix::<f32>::zeros(output_size, 1);
        EnvironmentalEncoder { weight_matrix, bias }
    }

    pub fn add_neurons(&self, rng: &mut ThreadRng, new_neurons: usize) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Create new weight rows for new neurons
        let extra_weights = DMatrix::<f32>::from_fn(new_neurons, self.weight_matrix.ncols(), |_, _| {
            normal.sample(rng)
        });

        // Create new bias entries for new neurons
        let extra_bias = DMatrix::<f32>::from_fn(new_neurons, 1, |_, _| {
            normal.sample(rng)
        });

        // Concatenate vertically (existing rows + new neurons)
        let new_weight_matrix = DMatrix::from_rows(
            &[
                self.weight_matrix.row_iter().collect::<Vec<_>>(),
                extra_weights.row_iter().collect::<Vec<_>>(),
            ]
            .concat(),
        );

        let new_bias = DMatrix::from_rows(
            &[
                self.bias.row_iter().collect::<Vec<_>>(),
                extra_bias.row_iter().collect::<Vec<_>>(),
            ]
            .concat(),
        );

        EnvironmentalEncoder {
            weight_matrix: new_weight_matrix,
            bias: new_bias,
        }
    }

    pub fn remove_neurons(&self, rng: &mut ThreadRng, num_remove: usize) -> Self {
        let total = self.weight_matrix.nrows();
        if num_remove >= total {
            panic!("Cannot remove all neurons from EnvironmentalEncoder");
        }

        // Randomly choose indices to keep
        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(rng);
        indices.truncate(total - num_remove);
        indices.sort_unstable();

        let new_weight_matrix = DMatrix::<f32>::from_rows(
            &indices.iter().map(|&i| self.weight_matrix.row(i)).collect::<Vec<_>>(),
        );

        let new_bias = DMatrix::<f32>::from_rows(
            &indices.iter().map(|&i| self.bias.row(i)).collect::<Vec<_>>(),
        );

        EnvironmentalEncoder {
            weight_matrix: new_weight_matrix,
            bias: new_bias,
        }
    }

    pub fn random(input_size: usize, output_size: usize, rng: &mut ThreadRng) -> Self {
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        let weight_matrix = DMatrix::<f32>::from_fn(output_size, input_size, |_, _| {
            normal.sample(rng)
        });
        
        let bias = DMatrix::<f32>::from_fn(output_size, 1, |_, _| {
            normal.sample(rng)
        });
        
        EnvironmentalEncoder { weight_matrix, bias }
    }


    pub fn mutate(&self, rng: &mut ThreadRng) -> Self{
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        let new_weights: DMatrix<f32> = self.weight_matrix.map(|x| {
            if rng.random_bool(MUTATION_RATE) {
                x + normal.sample(rng)
            } else {
                x
            }
        });

        let new_bias: DMatrix<f32> = self.bias.map(|x| {
            if rng.random_bool(MUTATION_RATE) {
                x + normal.sample(rng)
            } else {
                x
            }
        });

        EnvironmentalEncoder{weight_matrix: new_weights, bias: new_bias}
    }

    pub fn with_weights(weight_matrix: DMatrix<f32>, bias: DMatrix<f32>) -> Self {
        EnvironmentalEncoder { weight_matrix, bias }
    }

    pub fn encode(&self, input: &DMatrix<f32>) -> DMatrix<f32> {
        &self.weight_matrix * input + &self.bias
    }
}

pub struct CognitiveDecoder {
    pub(crate) weights: DVector<f32>,
}

impl CognitiveDecoder {
    pub fn new(input_size: usize) -> Self {
        let weights = DVector::<f32>::zeros(input_size);
        CognitiveDecoder { weights }
    }

    pub fn remove_neurons(&self, rng: &mut ThreadRng, num_remove: usize) -> Self {
        let total = self.weights.len();
        if num_remove >= total {
            panic!("Cannot remove all neurons from CognitiveDecoder");
        }

        let mut indices: Vec<usize> = (0..total).collect();
        indices.shuffle(rng);
        indices.truncate(total - num_remove);
        indices.sort_unstable();

        let new_weights = DVector::<f32>::from_iterator(
            indices.len(),
            indices.iter().map(|&i| self.weights[i]),
        );

        CognitiveDecoder { weights: new_weights }
    }

    pub fn with_weights(weights: DVector<f32>) -> Self {
        CognitiveDecoder { weights }
    }

    pub fn random(input_size: usize, rng: &mut ThreadRng) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let weights = DVector::from_fn(input_size, |_, _| {
            normal.sample(rng)
        });

        CognitiveDecoder { weights }
    }

    pub fn add_neurons(&self, rng: &mut ThreadRng, new_neurons: usize) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Create new weights for new neurons
        let extra_weights = DVector::<f32>::from_fn(new_neurons, |_, _| {
            normal.sample(rng)
        });

        // Concatenate the vectors
        let mut combined = DVector::<f32>::zeros(self.weights.len() + new_neurons);
        combined.rows_mut(0, self.weights.len()).copy_from(&self.weights);
        combined.rows_mut(self.weights.len(), new_neurons).copy_from(&extra_weights);

        CognitiveDecoder { weights: combined }
    }

    pub fn mutate(&self, rng: &mut ThreadRng) -> Self {
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        let new_weights: DVector<f32> = self.weights.map(|x| {
            if rng.random_bool(MUTATION_RATE) {
                x + normal.sample(rng)
            } else {
                x
            }
        });

        CognitiveDecoder { weights: new_weights }
    }

    pub fn decode(&self, input: &DVector<f32>) -> f32 {
        self.weights.dot(input)
    }
}
pub struct Cell {
    pub(crate) index: usize,
    brain_size: i32,
    current_energy: f32,
    state_encoder: EnvironmentalEncoder,
    social_decoder: CognitiveDecoder,
    hunger_decoder: CognitiveDecoder,
    isolation_decoder: CognitiveDecoder,
    metabolic_rate: f32,
    current_mass: f32,
    max_mass: f32,
    max_speed: f32, // NEW FIELD
    old_h: f32,
    old_iso: f32,
    old_soc: f32,
    old_encoding: DMatrix<f32>,
    sight_r: f32,
    sight_a: f32,
    desired_energy: f32,
    predation: f32
}

impl Cell {
    pub fn random(rng: &mut ThreadRng, mass: f32, index: usize) -> Self {
        let sight_r_dist: Normal<f32> = Normal::new(50.0, 5.0).unwrap();
        let sight_angle_dist: Normal<f32> = Normal::new(PI / 4.0, PI / 12.0).unwrap();
        let brain_size_dist: Normal<f32> = Normal::new(9.0, 2.0).unwrap();
        let predation_dist: Normal<f32> = Normal::new(0.5, 0.1).unwrap();

        let brain_size: i32 = brain_size_dist.sample(rng) as i32;
        let max_speed = rng.random_range(10.0..20.0); // NEW INITIALIZATION

        Cell {
            index,
            brain_size,
            current_energy: mass,
            state_encoder: EnvironmentalEncoder::random((65 + brain_size) as usize, brain_size as usize, rng),
            social_decoder: CognitiveDecoder::random(brain_size as usize, rng),
            hunger_decoder: CognitiveDecoder::random(brain_size as usize, rng),
            isolation_decoder: CognitiveDecoder::random(brain_size as usize, rng),
            metabolic_rate: 0.0,
            old_h: 0.0,
            old_iso: 0.0,
            old_soc: 0.0,
            old_encoding: DMatrix::<f32>::zeros((65 + brain_size) as usize, 1),
            current_mass: mass / 2.0,
            max_mass: mass,
            max_speed, // NEW FIELD
            sight_r: sight_r_dist.sample(rng),
            sight_a: sight_angle_dist.sample(rng),
            desired_energy: mass,
            predation: predation_dist.sample(rng),
        }
    }

    pub fn create_child(&self, world: &mut PhysicsSolver) {
        let child_pos: Vector2<f32> = world.positions[self.index] + Vector2::new(1.0, 0.0);
        let mut child_brain_size = self.brain_size;
        let mut child_mass = self.max_mass;
        let mut child_sight_r = self.sight_r;
        let mut child_sight_a = self.sight_a;
        let mut child_pred = self.predation;
        let mut child_max_speed = self.max_speed; // NEW FIELD COPY
        let mut brain_delta: i32 = 0;

        if world.rng.random_range(0.0..1.0) < (MUTATION_RATE / 2.0) {
            brain_delta = world.rng.random_range(-1..1);
            child_brain_size += brain_delta;
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_mass += world.rng.random_range(-1..1) as f32;
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_sight_r += world.rng.random_range(-1..1) as f32;
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_sight_a += world.rng.random_range(-0.05..0.05) as f32;
            child_sight_a = clamp(child_sight_a, 0.0, PI / 2.0);
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_pred += clamp(world.rng.random_range(-0.05..0.05) as f32, 0.0, 1.0);
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_max_speed += world.rng.random_range(-1.0..1.0); // NEW MUTATION
            child_max_speed = clamp(child_max_speed, 0.0, 50.0); // Prevent negative or absurd speeds
        }

        world.positions.push(child_pos);
        world.old_positions.push(child_pos);
        world.accelerations.push(Vector2::new(0.0, 0.0));
        world.masses.push(child_mass);
        world.radii.push(child_mass);
        world.is_plant.push(false);
        world.num_particles += 1;
        world.num_cells += 1;

        if brain_delta == 0 {
            world.cells.push(Cell {
                index: world.num_particles as usize,
                brain_size: child_brain_size,
                current_energy: child_mass,
                state_encoder: self.state_encoder.mutate(&mut world.rng),
                social_decoder: self.social_decoder.mutate(&mut world.rng),
                hunger_decoder: self.hunger_decoder.mutate(&mut world.rng),
                isolation_decoder: self.isolation_decoder.mutate(&mut world.rng),
                metabolic_rate: 0.0,
                current_mass: child_mass / 2.0,
                max_mass: child_mass,
                max_speed: child_max_speed, // NEW FIELD
                old_h: 0.0,
                old_iso: 0.0,
                old_soc: 0.0,
                old_encoding: DMatrix::<f32>::zeros((65 + self.brain_size) as usize, 1),
                sight_r: child_sight_r,
                sight_a: child_sight_a,
                desired_energy: child_mass,
                predation: child_pred,
            })
        } else if brain_delta > 0 {
            let child_se = self.state_encoder.add_neurons(&mut world.rng, brain_delta as usize).mutate(&mut world.rng);
            let child_sd = self.social_decoder.add_neurons(&mut world.rng, brain_delta as usize).mutate(&mut world.rng);
            let child_hd = self.hunger_decoder.add_neurons(&mut world.rng, brain_delta as usize).mutate(&mut world.rng);
            let child_id = self.isolation_decoder.add_neurons(&mut world.rng, brain_delta as usize).mutate(&mut world.rng);

            world.cells.push(Cell {
                index: world.num_particles as usize,
                brain_size: child_brain_size,
                current_energy: child_mass,
                state_encoder: child_se,
                social_decoder: child_sd,
                hunger_decoder: child_hd,
                isolation_decoder: child_id,
                old_h: 0.0,
                old_iso: 0.0,
                old_soc: 0.0,
                old_encoding: DMatrix::<f32>::zeros((65 + self.brain_size) as usize, 1),
                metabolic_rate: 0.0,
                current_mass: child_mass / 2.0,
                max_mass: child_mass,
                max_speed: child_max_speed, // NEW FIELD
                sight_r: child_sight_r,
                sight_a: child_sight_a,
                desired_energy: child_mass,
                predation: child_pred,
            })
        } else {
            brain_delta *= -1;
            let child_se = self.state_encoder.remove_neurons(&mut world.rng, brain_delta as usize).mutate(&mut world.rng);
            let child_sd = self.social_decoder.remove_neurons(&mut world.rng, brain_delta as usize).mutate(&mut world.rng);
            let child_hd = self.hunger_decoder.remove_neurons(&mut world.rng, brain_delta as usize).mutate(&mut world.rng);
            let child_id = self.isolation_decoder.remove_neurons(&mut world.rng, brain_delta as usize).mutate(&mut world.rng);

            world.cells.push(Cell {
                index: world.num_particles as usize,
                brain_size: child_brain_size,
                current_energy: child_mass,
                state_encoder: child_se,
                social_decoder: child_sd,
                hunger_decoder: child_hd,
                isolation_decoder: child_id,
                metabolic_rate: 0.0,
                current_mass: child_mass / 2.0,
                max_mass: child_mass,
                max_speed: child_max_speed, // NEW FIELD
                sight_r: child_sight_r,
                sight_a: child_sight_a,
                old_h: 0.0,
                old_iso: 0.0,
                old_soc: 0.0,
                old_encoding: DMatrix::<f32>::zeros((65 + self.brain_size) as usize, 1),
                desired_energy: child_mass,
                predation: child_pred,
            })
        }
    }

    pub fn encode_environment(&self, world: &PhysicsSolver) -> DMatrix<f32> {
        let pos: Vector2<f32> = world.positions[self.index];
        let vel: Vector2<f32> = world.positions[self.index] - world.old_positions[self.index];
        let point_idx: Vec<i32> = world.qt.query_cone(pos, vel, self.sight_r, self.sight_a);
        let mut full_env = DMatrix::<f32>::zeros((self.brain_size + 66) as usize, 1);

        for (i, &idx) in point_idx.iter().enumerate() {
            let other_pos = world.positions[idx as usize];
            let dx = (pos.x - other_pos.x) / self.sight_r;
            let dy = (pos.y - other_pos.y) / self.sight_r;

            let base = i * 3;
            full_env[(base, 0)] = dx;
            full_env[(base + 1, 0)] = dy;
            full_env[(base + 2, 0)] = world.is_plant[idx as usize] as i32 as f32;
        }

        full_env[60] = self.old_h;
        full_env[61] = self.old_iso;
        full_env[62] = self.old_soc;
        full_env[63] = self.desired_energy - self.current_energy;
        full_env[64] = self.metabolic_rate;

        full_env
            .view_mut((self.brain_size as usize, 0), (self.brain_size as usize, 1))
            .copy_from(&self.old_encoding);

        self.state_encoder.encode(&full_env)
    }

    pub fn timestep(&mut self, world: &mut PhysicsSolver) {
        let internal_rep = self.encode_environment(world);
        self.old_encoding = internal_rep.clone();
        self.old_h = self.hunger_decoder.decode(&internal_rep.column(0).into_owned());
        self.old_iso = self.isolation_decoder.decode(&internal_rep.column(0).into_owned());
        self.old_soc = self.social_decoder.decode(&internal_rep.column(0).into_owned());
    
        let pos: Vector2<f32> = world.positions[self.index];
        let vel: Vector2<f32> = world.positions[self.index] - world.old_positions[self.index];
        let point_idx: Vec<i32> = world.qt.query_cone(pos, vel, self.sight_r, self.sight_a);
       
        // Check for eating opportunities first
        let eating_range = world.radii[self.index] * 1.2; // Slightly larger than collision radius
        for &idx in &point_idx {
            if idx == self.index as i32 {
                continue;
            }
            
            let target_pos = world.positions[idx as usize];
            let distance = (pos - target_pos).magnitude();
            
            if distance <= eating_range {
                let is_plant = world.is_plant[idx as usize];
                let can_eat_animal = self.predation > 0.5 && !is_plant;
                
                if is_plant || can_eat_animal {
                    // Calculate energy gain (with some efficiency loss)
                    let energy_gain = world.masses[idx as usize] * 0.8;
                    self.current_energy += energy_gain;
                    
                    // Mark the eaten entity for deletion
                    // Note: You'll need to handle the actual deletion in PhysicsSolver
                    // to avoid index issues during iteration
                    world.delete_particle(idx as usize);
                }
            }
        }
    
        let food_points: Vec<Vector2<f32>> = point_idx.iter()
            .filter(|&&idx| world.is_plant[idx as usize] || (self.predation > 0.5) && idx != self.index as i32)
            .map(|&idx| world.positions[idx as usize])
            .collect();
    
        let mut social_points: Vec<Vector2<f32>> = point_idx.iter()
            .filter(|&&idx| !world.is_plant[idx as usize] && idx != self.index as i32)
            .map(|&idx| world.positions[idx as usize])
            .collect();
        
        let mean_social_point = Vector2::new(
            social_points.iter().map(|p| p.x).sum::<f32>() / social_points.len() as f32,
            social_points.iter().map(|p| p.y).sum::<f32>() / social_points.len() as f32,
        );
    
        let mean_hunger_point = Vector2::new(
            food_points.iter().map(|p| p.x).sum::<f32>() / food_points.len() as f32,
            food_points.iter().map(|p| p.y).sum::<f32>() / food_points.len() as f32,
        ); 
    
        let hunger_vector = mean_hunger_point - pos;
        let social_vector = mean_social_point - pos;
        let isolation_vector = social_vector * -1.0;
    
        let hunger_vel = hunger_vector.normalize() * self.max_speed;
        let social_vel = social_vector.normalize() * self.max_speed;
        let isolation_vel = isolation_vector.normalize() * self.max_speed;
    
        let max_motivation = self.old_h + self.old_iso + self.old_soc;
        let hunger_motivation = self.old_h / max_motivation;
        let social_motivation = self.old_soc / max_motivation;
        let isolation_motivation = self.old_iso / max_motivation;
    
        let mut acc = hunger_vel * hunger_motivation
            + social_vel * social_motivation
            + isolation_vel * isolation_motivation;
        world.accelerate_particle(self.index, acc);
    
        self.metabolic_rate = f32::powf(self.current_mass, 0.75);
        self.metabolic_rate += f32::powf(self.brain_size as f32, 0.86);
        self.metabolic_rate += f32::powf(self.current_mass as f32, 0.7) * acc.magnitude() / 100.0;
    
        self.current_energy -= self.metabolic_rate * 0.0167;
    
        if self.current_energy <= self.desired_energy{
            self.current_energy += self.current_mass * 0.01;
            self.current_mass -= self.current_mass * 0.01;
        }else if self.current_energy > self.desired_energy {
            self.current_mass += self.current_mass * 0.01;
            if self.current_mass > self.max_mass {
                self.current_mass = self.max_mass;
                self.create_child(world);
                self.current_mass = self.max_mass / 0.75;
            }
        }
    }
}
