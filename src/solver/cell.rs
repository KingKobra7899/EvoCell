use nalgebra::{DMatrix};
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

    pub fn decode(&self, input: &DMatrix<f32>) -> f32 {
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