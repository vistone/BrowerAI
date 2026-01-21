use serde::{Deserialize, Serialize};
use std::any::Any;

pub trait AiModel: Send + Sync {
    fn infer(&self, input: &Tensor) -> Result<Tensor, AiInferenceError>;

    fn batch_infer(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, AiInferenceError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.infer(input)?);
        }
        Ok(results)
    }

    fn get_metadata(&self) -> ModelMetadata;

    fn get_input_spec(&self) -> &[TensorSpec];
    fn get_output_spec(&self) -> &[TensorSpec];

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: DataType,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float16,
    Int64,
    Int32,
    Uint8,
    Bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DataType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub description: String,
    pub parameters: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelType {
    Transformer,
    Cnn,
    Rnn,
    Mlp,
    CodePredictor,
    Custom,
}

#[derive(Debug)]
pub struct AiInferenceError {
    pub message: String,
    pub model: String,
    pub recoverable: bool,
    pub inference_time_ms: u64,
}

impl std::fmt::Display for AiInferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AI inference error in {}: {}", self.model, self.message)?;
        if !self.recoverable {
            write!(f, " (non-recoverable)")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={:?}, dtype={:?})", self.shape, self.dtype)
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            dtype: DataType::Float32,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let expected: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(expected, new_size, "Tensor reshape size mismatch");
        self.shape = new_shape;
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            shape: Vec::new(),
            dtype: DataType::Float32,
        }
    }
}

pub struct TensorBuilder {
    data: Vec<f32>,
    shape: Vec<usize>,
    dtype: DataType,
}

impl TensorBuilder {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            shape: Vec::new(),
            dtype: DataType::Float32,
        }
    }

    pub fn with_data(mut self, data: Vec<f32>) -> Self {
        self.data = data;
        self
    }

    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.shape = shape;
        self
    }

    pub fn with_dtype(mut self, dtype: DataType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn build(self) -> Tensor {
        Tensor {
            data: self.data,
            shape: self.shape,
            dtype: self.dtype,
        }
    }
}

impl Default for TensorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        assert_eq!(tensor.len(), 3);
        assert_eq!(tensor.shape, vec![1, 3]);
    }

    #[test]
    fn test_tensor_reshape() {
        let mut tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        tensor.reshape(vec![4, 1]);
        assert_eq!(tensor.shape, vec![4, 1]);
    }

    #[test]
    fn test_tensor_builder() {
        let tensor = TensorBuilder::new()
            .with_data(vec![1.0, 2.0])
            .with_shape(vec![2])
            .with_dtype(DataType::Float32)
            .build();
        assert_eq!(tensor.len(), 2);
    }

    #[test]
    fn test_model_metadata() {
        let metadata = ModelMetadata {
            name: "test-model".to_string(),
            version: "1.0.0".to_string(),
            model_type: ModelType::Transformer,
            description: "Test model".to_string(),
            parameters: 1000000,
        };
        assert_eq!(metadata.name, "test-model");
    }

    #[test]
    fn test_inference_error() {
        let error = AiInferenceError {
            message: "Test error".to_string(),
            model: "test-model".to_string(),
            recoverable: true,
            inference_time_ms: 100,
        };
        assert!(error.to_string().contains("test-model"));
        assert!(error.recoverable);
    }
}
