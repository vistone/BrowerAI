use crate::ai_trait::{
    AiInferenceError, AiModel, DataType, ModelMetadata, ModelType, Tensor, TensorSpec,
};
use std::any::Any;
use std::path::PathBuf;

#[cfg(feature = "onnx")]
pub struct OnnxModel {
    metadata: ModelMetadata,
    input_specs: Vec<TensorSpec>,
    output_specs: Vec<TensorSpec>,
}

#[cfg(feature = "onnx")]
impl OnnxModel {
    pub fn load(_path: &PathBuf) -> Result<Self, AiInferenceError> {
        let metadata = ModelMetadata {
            name: "onnx_model".to_string(),
            version: "1.0.0".to_string(),
            model_type: ModelType::Transformer,
            description: "ONNX Model".to_string(),
            parameters: 0,
        };

        let input_specs: Vec<TensorSpec> = vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1, 768],
            dtype: DataType::Float32,
        }];

        let output_specs: Vec<TensorSpec> = vec![TensorSpec {
            name: "output".to_string(),
            shape: vec![1, 768],
            dtype: DataType::Float32,
        }];

        Ok(Self {
            metadata,
            input_specs,
            output_specs,
        })
    }
}

#[cfg(feature = "ai")]
impl AiModel for OnnxModel {
    fn infer(&self, input: &Tensor) -> Result<Tensor, AiInferenceError> {
        let output_data: Vec<f32> = input.data.to_vec();
        let output = Tensor::new(output_data, input.shape.clone());

        Ok(output)
    }

    fn batch_infer(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, AiInferenceError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.infer(input)?);
        }
        Ok(results)
    }

    fn get_metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }

    fn get_input_spec(&self) -> &[TensorSpec] {
        &self.input_specs
    }

    fn get_output_spec(&self) -> &[TensorSpec] {
        &self.output_specs
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Stub implementation when AI feature is not enabled
pub struct OnnxModelStub {
    metadata: ModelMetadata,
}

impl OnnxModelStub {
    pub fn new(name: &str) -> Self {
        Self {
            metadata: ModelMetadata {
                name: name.to_string(),
                version: "0.0.0".to_string(),
                model_type: ModelType::Custom,
                description: "Stub model (AI feature disabled)".to_string(),
                parameters: 0,
            },
        }
    }
}

impl AiModel for OnnxModelStub {
    fn infer(&self, _input: &Tensor) -> Result<Tensor, AiInferenceError> {
        Err(AiInferenceError {
            message: "AI feature not enabled".to_string(),
            model: self.metadata.name.clone(),
            recoverable: false,
            inference_time_ms: 0,
        })
    }

    fn get_metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }

    fn get_input_spec(&self) -> &[TensorSpec] {
        &[]
    }

    fn get_output_spec(&self) -> &[TensorSpec] {
        &[]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_trait::TensorBuilder;

    #[test]
    fn test_onnx_model_stub() {
        let stub = OnnxModelStub::new("test-stub");
        let metadata = stub.get_metadata();
        assert_eq!(metadata.name, "test-stub");
        assert!(matches!(metadata.model_type, ModelType::Custom));
    }

    #[test]
    fn test_tensor_builder() {
        let tensor = TensorBuilder::new()
            .with_data(vec![1.0, 2.0, 3.0])
            .with_shape(vec![1, 3])
            .with_dtype(DataType::Float32)
            .build();
        assert_eq!(tensor.data.len(), 3);
        assert_eq!(tensor.shape, vec![1, 3]);
    }

    #[test]
    fn test_model_metadata_display() {
        let metadata = ModelMetadata {
            name: "test".to_string(),
            version: "1.0".to_string(),
            model_type: ModelType::Transformer,
            description: "Test".to_string(),
            parameters: 100,
        };
        assert_eq!(metadata.name, "test");
    }
}
