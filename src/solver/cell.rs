use std::f32::consts::PI;

use nalgebra::{DMatrix, DVector};
use rand::{rngs::ThreadRng, Rng};
use rand_distr::{Normal, Distribution};

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


    pub fn mutate(&self, rng: &mut ThreadRng){
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

        EnvironmentalEncoder{weight_matrix: new_weights, bias: new_bias};
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
pub struct Cell{
    pub(crate) index: usize,
    brain_size: i32,
    current_energy: f32,
    state_encoder: EnvironmentalEncoder,
    social_decoder:  CognitiveDecoder,
    hunger_decoder: CognitiveDecoder,
    isolation_decoder: CognitiveDecoder,
    gr_decoder: CognitiveDecoder, //growth converts energy to mass according to the function
    current_mass: f32,
    max_mass: f32,
    sight_r: f32,
    sight_a: f32,
    desired_energy: f32,
    predation: f32
}

impl Cell{
    pub fn random(rng: &mut ThreadRng, mass: f32, index: usize) -> Self {
        
        let sight_r_dist:Normal<f32> = Normal::new(50.0, 5.0).unwrap();
        let sight_angle_dist:Normal<f32> = Normal::new(PI / 4.0, PI / 12.0).unwrap();
        let brain_size_dist:Normal<f32>  = Normal::new(9.0, 2.0).unwrap();
        let predation_dist:Normal<f32> = Normal::new(0.5, 0.1).unwrap();

       
        let brain_size: i32 = brain_size_dist.sample(rng) as i32;

        // max of 20 creatures (encoded as the vec btwn them)
        // 40 inputs
        // 4 (activation of each decoder)
        // caloric balance (desired - current)
        // current metabolic rate
        // brain_size (past activation)

        //final brain input size is 40 + 4 + 1 + 1 + brain_size
        //46 + brain_size

        Cell { index: index,
             brain_size: brain_size,
             current_energy: mass, 
             state_encoder: EnvironmentalEncoder::random((46 + brain_size) as usize, brain_size as usize, rng), 
             social_decoder: CognitiveDecoder::random(brain_size as usize, rng), 
             hunger_decoder: CognitiveDecoder::random(brain_size as usize, rng),  
             isolation_decoder: CognitiveDecoder::random(brain_size as usize, rng),  
             gr_decoder: CognitiveDecoder::random(brain_size as usize, rng), 
             current_mass: mass / 2.0, 
             max_mass: mass, 
             sight_r: sight_r_dist.sample(rng), 
             sight_a: sight_angle_dist.sample(rng), 
             desired_energy: mass,
             predation: predation_dist.sample(rng)
            }
    }


}