use nalgebra::{DMatrix};
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
    pub(crate) weights: DMatrix<f32>,
}

impl CognitiveDecoder {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = DMatrix::<f32>::zeros(output_size, input_size);
        CognitiveDecoder { weights }
    }

    pub fn with_weights(weights: DMatrix<f32>) -> Self {
        CognitiveDecoder { weights }
    }

    pub fn mutate(&self, rng: &mut ThreadRng) -> Self {
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        let new_weights: DMatrix<f32> = self.weights.map(|x| {
            if rng.random_bool(MUTATION_RATE) {
                x + normal.sample(rng)
            } else {
                x
            }
        });

        CognitiveDecoder { weights: new_weights }
    }

    pub fn decode(&self, input: &DMatrix<f32>) -> f32 {
        self.weights.dot(input)
    }
}
pub struct Cell{
    index: usize,
    brain_size: i32,
    current_energy: f32,
    state_encoder: EnvironmentalEncoder,
    social_decoder:  CognitiveDecoder,
    hunger_decoder: CognitiveDecoder,
    isolation_decoder: CognitiveDecoder,
    gr_decoder: CognitiveDecoder, //growth converts energy to mass according to the function
    desired_mass: f32,
    desired_energy: f32
}