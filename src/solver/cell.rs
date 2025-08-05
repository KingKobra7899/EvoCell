// cell.rs - Fixed version
use std::f32::consts::PI;

use nalgebra::{clamp, DMatrix, DVector, Vector2, VectorN};
use rand::{random_range, rngs::ThreadRng, seq::SliceRandom as _, Rng};
use rand_distr::{Normal, Distribution};
use nalgebra_glm as glm;
use crate::solver::{quadtree::Rect, PhysicsSolver};

const MUTATION_RATE: f64 = 0.1;

use serde::{Serialize, Deserialize};

#[derive(Serialize)]
struct BrainExport {
    brain_size: i32,
    encoder_weights: Vec<f32>,
    encoder_bias: Vec<f32>,
    social_weights: Vec<f32>,
    hunger_weights: Vec<f32>,
    isolation_weights: Vec<f32>,
    sight_r: f32,
    sight_a: f32,
    predation: f32,
    max_speed: f32
}

pub struct EnvironmentalEncoder {
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

        let extra_weights = DMatrix::<f32>::from_fn(new_neurons, self.weight_matrix.ncols(), |_, _| {
            normal.sample(rng)
        });

        let extra_bias = DMatrix::<f32>::from_fn(new_neurons, 1, |_, _| {
            normal.sample(rng)
        });

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

    pub fn mutate(&self, rng: &mut ThreadRng) -> Self {
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

        EnvironmentalEncoder { weight_matrix: new_weights, bias: new_bias }
    }

    pub fn with_weights(weight_matrix: DMatrix<f32>, bias: DMatrix<f32>) -> Self {
        EnvironmentalEncoder { weight_matrix, bias }
    }

    pub fn encode(&self, input: &DMatrix<f32>) -> DMatrix<f32> {
        DMatrix::map(&(self.weight_matrix.clone() * input + &self.bias), |x| x.tanh())
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

        let extra_weights = DVector::<f32>::from_fn(new_neurons, |_, _| {
            normal.sample(rng)
        });

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
        f32::tanh(self.weights.dot(input)) + 1.0 // Ensure output is positive
    }
}

pub struct Cell {
    pub(crate) index: usize,
    pub(crate) brain_size: i32,
    pub(crate) current_energy: f32,
    pub(crate) state_encoder: EnvironmentalEncoder,
    pub social_decoder: CognitiveDecoder,
    pub hunger_decoder: CognitiveDecoder,
    pub isolation_decoder: CognitiveDecoder,
    pub metabolic_rate: f32,
    pub current_mass: f32,
    pub max_mass: f32,
    pub max_speed: f32,
    pub old_h: f32,
    pub old_iso: f32,
    pub old_soc: f32,
    pub old_encoding: DMatrix<f32>,
    pub sight_r: f32,
    pub sight_a: f32,
    pub desired_energy: f32,
    pub predation: f32,
    pub to_delete: bool, // New field to mark for deletion
}

impl Cell {
    pub fn random(rng: &mut ThreadRng, mass: f32, index: usize) -> Self {
        let sight_r_dist: Normal<f32> = Normal::new(50.0, 10.0).unwrap();
        let sight_angle_dist: Normal<f32> = Normal::new(PI / 4.0, PI / 12.0).unwrap();
        let brain_size_dist: Normal<f32> = Normal::new(9.0, 2.0).unwrap();
        let predation_dist: Normal<f32> = Normal::new(0.5, 0.1).unwrap();

        let brain_size: i32 = brain_size_dist.sample(rng) as i32;
        let max_speed = rng.random_range(0.1..0.5);

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
            old_encoding: DMatrix::<f32>::zeros((brain_size) as usize, 1),
            current_mass: mass / 2.0,
            max_mass: mass,
            max_speed: max_speed,
            sight_r: sight_r_dist.sample(rng),
            sight_a: sight_angle_dist.sample(rng),
            desired_energy: mass,
            predation: predation_dist.sample(rng),
            to_delete: false,
        }
    }

    pub fn create_child(&self, world: &mut PhysicsSolver) {
        let child_pos: Vector2<f32> = world.positions[self.index] + Vector2::new(world.radii[self.index] * 4.3, 0.0);
        let mut child_brain_size = self.brain_size;
        let mut child_mass = self.max_mass;
        let mut child_sight_r = self.sight_r;
        let mut child_sight_a = self.sight_a;
        let mut child_pred = self.predation;
        let mut child_max_speed = self.max_speed;
        let mut brain_delta: i32 = 0;

        if world.rng.random_range(0.0..1.0) < (MUTATION_RATE / 2.0) {
            //brain_delta = world.rng.random_range(-1..1);
            child_brain_size += brain_delta;
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_mass += clamp(world.rng.random_range(-1..1) as f32, 5.0, 100.0);
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_sight_r += world.rng.random_range(-0.1..0.1) as f32;
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_sight_a += world.rng.random_range(-0.05..0.05) as f32;
            child_sight_a = clamp(child_sight_a, 0.0, PI / 2.0);
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_pred += clamp(world.rng.random_range(-0.01..0.01) as f32, 0.0, 1.0);
        }
        if world.rng.random_range(0.0..1.0) < MUTATION_RATE {
            child_max_speed += world.rng.random_range(-1.0..1.0);
            child_max_speed = clamp(child_max_speed, 0.0, 50.0);
        }

        // Add child to pending additions instead of directly adding
        world.pending_additions.push((child_pos, child_mass, brain_delta, child_brain_size, clamp(child_sight_r,0.0, 100.0), child_sight_a, child_pred, child_max_speed, self.clone()));
    }

    pub fn encode_environment(&self, world: &PhysicsSolver) -> DMatrix<f32> {
        let pos: Vector2<f32> = world.positions[self.index];
        let vel: Vector2<f32> = world.positions[self.index] - world.old_positions[self.index];
        let point_idx: Vec<i32> = world.qt.query_cone(pos, vel, self.sight_r, self.sight_a);
        let mut full_env = DMatrix::<f32>::zeros((self.brain_size + 65) as usize, 1);

        for (i, &idx) in point_idx.iter().enumerate() {
            if i >= 20 { break; } // Limit to prevent overflow
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

        for val in full_env.iter_mut() {
            if !val.is_finite() {
                *val = 0.0;
            }
        }

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

        // Eating logic - mark particles for deletion instead of deleting immediately
        let eating_range = world.radii[self.index] * 1.0;
        let eating_points = world.qt.query(&Rect::new(pos.x, pos.y, world.radii[self.index] * 5.0, world.radii[self.index] * 5.0));

        for &idx in &eating_points {
            if idx == self.index as i32 {
                continue;
            }

            let target_pos = world.positions[idx as usize];
            let distance = (pos - target_pos).magnitude();

            if distance <= (eating_range + world.radii[idx as usize]) {
                let is_plant = world.is_plant[idx as usize];
                let can_eat_animal = self.predation > 0.5 && !is_plant;

                if is_plant || (can_eat_animal && idx != self.index as i32) && world.masses[idx as usize] < self.current_mass {
                if(world.rng.random_range(0.0..1.0) < 0.1) { // 50% chance to successfully eat
                    let energy_gain = world.masses[idx as usize] * 20.0; // 20x mass to energy conversion
                    self.current_energy += energy_gain;
                    //println!("Cell {} ate {} at distance {}", self.index, idx, distance);
                    
                    // Mark for deletion instead of deleting immediately
                    world.pending_deletions.push(idx as usize);
                    break;
                }
                }else if !can_eat_animal && !is_plant {
                    self.current_mass -= 0.01 * self.current_mass; // Lose mass if can't eat
                    
                    //increase mass of other creature (sharing!)
                    for cell in world.cells.iter_mut() {
                        if cell.index == idx as usize {
                            cell.current_mass += 0.01 * self.current_mass; // Share some mass
                            break;
                        }
                    }
                }
            }
        }

        // Movement logic (unchanged)
        let food_points: Vec<Vector2<f32>> = point_idx.iter()
            .filter(|&&idx| {
                world.is_plant[idx as usize]
                    || (self.predation > 0.5 && idx != self.index as i32)
            })
            .map(|&idx| world.positions[idx as usize])
            .collect();

        let social_points: Vec<Vector2<f32>> = point_idx.iter()
            .filter(|&&idx| !world.is_plant[idx as usize] && idx != self.index as i32)
            .map(|&idx| world.positions[idx as usize])
            .collect();

        let mean_social_point = if !social_points.is_empty() {
            Vector2::new(
                social_points.iter().map(|p| p.x).sum::<f32>() / social_points.len() as f32,
                social_points.iter().map(|p| p.y).sum::<f32>() / social_points.len() as f32,
            )
        } else {
            pos
        };

        let mean_hunger_point = if !food_points.is_empty() {
            Vector2::new(
                food_points.iter().map(|p| p.x).sum::<f32>() / food_points.len() as f32,
                food_points.iter().map(|p| p.y).sum::<f32>() / food_points.len() as f32,
            )
        } else {
            pos
        };

        let safe_normalize = |v: Vector2<f32>| {
            let mag = v.magnitude();
            if mag > 0.0 { v / mag } else { Vector2::new(0.0, 0.0) }
        };

        let hunger_vector = mean_hunger_point - pos;
        let social_vector = mean_social_point - pos;
        let isolation_vector = social_vector * -1.0;

        let hunger_vel = if hunger_vector.magnitude() > self.max_speed { safe_normalize(hunger_vector) * self.max_speed } else { hunger_vector };
        let social_vel = if social_vector.magnitude() > self.max_speed { safe_normalize(social_vector) * self.max_speed } else { social_vector };
        let isolation_vel = if isolation_vector.magnitude() > self.max_speed { safe_normalize(isolation_vector) * self.max_speed } else { isolation_vector };

        let max_motivation = self.old_h + self.old_soc + self.old_iso;
        let (hunger_motivation, social_motivation, isolation_motivation) =
            if max_motivation.abs() > f32::EPSILON {
                (
                    self.old_h / max_motivation,
                    self.old_soc / max_motivation,
                    self.old_iso / max_motivation,
                )
            } else {
                // Equal probability when all motivations are zero
                (1.0/3.0, 1.0/3.0, 1.0/3.0)
            };

            
            let mut movement = hunger_vel * hunger_motivation
                + social_vel * social_motivation
                + isolation_vel * isolation_motivation;

            if movement.x.is_nan() || movement.y.is_nan() {
                
                
                // Random direction in radians
                let angle = world.rng.random_range(0.0..std::f32::consts::TAU);
                
                // Random speed <= max_speed
                let speed = world.rng.random_range(0.0..=self.max_speed);

                movement = Vector2::new(angle.cos(), angle.sin()) * speed;
            }

            world.positions[self.index] += movement;

        // ----- Metabolic calculations -----
        let basal_cost = 0.02 * f32::powf(self.current_mass, 0.75); // Kleiber's law
        let brain_cost = 0.025* f32::powf(self.brain_size as f32, 0.86);
        let move_cost = 0.05 * self.current_mass * movement.magnitude().powi(2); // cost grows quadratically with acceleration
       
        self.metabolic_rate = basal_cost + brain_cost + move_cost;
        self.current_energy -= self.metabolic_rate;

        // ----- Energy ↔ Mass exchange -----
        let starvation_threshold = self.desired_energy * 0.5;
        if self.current_energy < starvation_threshold {
            // Burn mass for energy
            let mass_loss = 0.002 * self.current_mass;
            self.current_mass -= mass_loss;
            self.current_energy += mass_loss * 20.0; // conversion rate from mass to energy
        } else {
            // Store surplus energy as mass
            let storage = 0.001 * self.current_energy;
            self.current_mass += storage;
            self.current_energy -= storage;
        }

        // Cap mass within bounds
        if self.current_mass > self.max_mass {
            self.current_mass = self.max_mass;
            self.create_child(world);
            self.current_energy *= 0.5; // post-reproduction energy loss
            self.current_mass *= 0.5; // post-reproduction weight loss
        }
        
        world.masses[self.index] = self.current_mass;
        world.avg_speed += self.max_speed / world.num_cells as f32;
        world.avg_brain_size += self.brain_size as f32 / world.num_cells as f32;
        world.avg_hunger +=if self.old_h.is_finite() {self.old_h / world.num_cells as f32} else {0.0};
        world.avg_isolation += if self.old_iso.is_finite() {self.old_iso /world.num_cells as f32} else {0.0};
        world.avg_social += if self.old_soc.is_finite() {self.old_soc / world.num_cells as f32} else {0.0};
        world.avg_sight_r += self.sight_r / world.num_cells as f32;


        // ----- Death conditions -----
        let min_mass_for_survival = 0.05; // scaleable for realism
        if self.current_energy <= 0.0 || self.current_mass <= min_mass_for_survival || world.positions[self.index].x.is_nan() || world.positions[self.index].y.is_nan() {
            self.to_delete = true;
            return;
        }

    }
    pub fn export_brain(&self) -> BrainExport {
        BrainExport {
            brain_size: self.brain_size,
            encoder_weights: self.state_encoder.weight_matrix.as_slice().to_vec(),
            encoder_bias: self.state_encoder.bias.as_slice().to_vec(),
            social_weights: self.social_decoder.weights.as_slice().to_vec(),
            hunger_weights: self.hunger_decoder.weights.as_slice().to_vec(),
            isolation_weights: self.isolation_decoder.weights.as_slice().to_vec(),
            sight_r: self.sight_r,
            sight_a: self.sight_a,
            predation: self.predation,
            max_speed: self.max_speed,
        }
    }

    /// Export brain to JSON string
    pub fn export_brain_json(&self) -> Result<String, serde_json::Error> {
        let brain_export = self.export_brain();
        serde_json::to_string_pretty(&brain_export)
    }

    /// Save brain to file
    pub fn save_brain_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.export_brain_json()?;
        let mut file = std::fs::File::create(filename)?;
        std::io::Write::write_all(&mut file, json.as_bytes())?;
        Ok(())
    }
}

impl Clone for Cell {
    fn clone(&self) -> Self {
        Cell {
            index: self.index,
            brain_size: self.brain_size,
            current_energy: self.current_energy,
            state_encoder: EnvironmentalEncoder::with_weights(self.state_encoder.weight_matrix.clone(), self.state_encoder.bias.clone()),
            social_decoder: CognitiveDecoder::with_weights(self.social_decoder.weights.clone()),
            hunger_decoder: CognitiveDecoder::with_weights(self.hunger_decoder.weights.clone()),
            isolation_decoder: CognitiveDecoder::with_weights(self.isolation_decoder.weights.clone()),
            metabolic_rate: self.metabolic_rate,
            current_mass: self.current_mass,
            max_mass: self.max_mass,
            max_speed: self.max_speed,
            old_h: self.old_h,
            old_iso: self.old_iso,
            old_soc: self.old_soc,
            old_encoding: self.old_encoding.clone(),
            sight_r: self.sight_r,
            sight_a: self.sight_a,
            desired_energy: self.desired_energy,
            predation: self.predation,
            to_delete: self.to_delete,
        }
    }
}