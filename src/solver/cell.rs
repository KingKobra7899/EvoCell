use nalgebra::{matrix, Matrix};

pub struct EnvironmentalEncoder{
    pub(crate) weight_matrix: Matrix<f32>,
    pub(crate) bias: Matrix<f32>,
}

impl EnvironmentalEncoder {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weight_matrix = matrix![0.0; output_size, input_size];
        let bias = matrix![0.0; output_size, 1];
        EnvironmentalEncoder { weight_matrix, bias }
    }

    pub fn encode(&self, input: &Matrix<f32>) -> Matrix<f32> {
        self.weight_matrix * input + &self.bias
    }
}

pub struct CognitiveDecoder {
    pub(crate) weights: Matrix<f32>,
}

impl CognitiveDecoder {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights: Matrix<f32, _, _, _> = matrix![0.0; output_size, input_size];
        CognitiveDecoder { weights }
    }

    pub fn decode(&self, input: &Matrix<f32>) -> Matrix<f32> {
        self.weights.dot(input)
    }
}
pub struct Cell{
    index: i32,
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