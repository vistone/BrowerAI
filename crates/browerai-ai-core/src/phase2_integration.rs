//! Phase 2 模型集成模块
//!
//! 处理 5 个 ONNX 模型的推理：
//! 1. SelectorEmbedding - CSS 选择器嵌入 (2.83M)
//! 2. PropertyPredictor - CSS 属性预测 (2.66M)
//! 3. ColorLearning - 颜色特征提取 (4.40M)
//! 4. CompletePageModel - 统一页面表示 (1.65M)
//! 5. FinetunedModel - LoRA 微调 (0.27M)

use anyhow::{anyhow, Context, Result};
use log;
use std::path::Path;

#[cfg(feature = "onnx")]
use ort::{session::Session, value::Tensor as OrtTensor};

/// Phase 2 ONNX 模型加载器
pub struct Phase2ModelLoader {
    model_dir: std::path::PathBuf,
}

impl Phase2ModelLoader {
    /// 创建新的模型加载器
    pub fn new(model_dir: impl AsRef<Path>) -> Self {
        Self {
            model_dir: model_dir.as_ref().to_path_buf(),
        }
    }

    /// 加载选择器嵌入模型
    #[cfg(feature = "onnx")]
    pub fn load_selector_embedding(&self) -> Result<Phase2SelectorEmbedding> {
        let path = self.model_dir.join("onnx_exports/selector_embedding.onnx");
        log::info!("Loading selector_embedding model from {:?}", path);

        if !path.exists() {
            return Err(anyhow!("Model file not found: {:?}", path));
        }

        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(&path)
            .context("Failed to load selector_embedding ONNX model")?;

        Ok(Phase2SelectorEmbedding {
            session: std::sync::Arc::new(session),
            model_path: path,
        })
    }

    /// 加载属性预测模型
    #[cfg(feature = "onnx")]
    pub fn load_property_predictor(&self) -> Result<Phase2PropertyPredictor> {
        let path = self.model_dir.join("onnx_exports/property_predictor.onnx");
        log::info!("Loading property_predictor model from {:?}", path);

        if !path.exists() {
            return Err(anyhow!("Model file not found: {:?}", path));
        }

        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(&path)
            .context("Failed to load property_predictor ONNX model")?;

        Ok(Phase2PropertyPredictor {
            session: std::sync::Arc::new(session),
            model_path: path,
        })
    }

    /// 加载颜色学习模型
    #[cfg(feature = "onnx")]
    pub fn load_color_model(&self) -> Result<Phase2ColorModel> {
        let path = self.model_dir.join("onnx_exports/color_model.onnx");
        log::info!("Loading color_model from {:?}", path);

        if !path.exists() {
            return Err(anyhow!("Model file not found: {:?}", path));
        }

        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(&path)
            .context("Failed to load color_model ONNX model")?;

        Ok(Phase2ColorModel {
            session: std::sync::Arc::new(session),
            model_path: path,
        })
    }

    /// 加载完整页面模型
    #[cfg(feature = "onnx")]
    pub fn load_complete_model(&self) -> Result<Phase2CompleteModel> {
        let path = self.model_dir.join("onnx_exports/complete_model.onnx");
        log::info!("Loading complete_model from {:?}", path);

        if !path.exists() {
            return Err(anyhow!("Model file not found: {:?}", path));
        }

        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(&path)
            .context("Failed to load complete_model ONNX model")?;

        Ok(Phase2CompleteModel {
            session: std::sync::Arc::new(session),
            model_path: path,
        })
    }

    /// 加载微调模型
    #[cfg(feature = "onnx")]
    pub fn load_finetuned_model(&self) -> Result<Phase2FinetunedModel> {
        let path = self.model_dir.join("onnx_exports/finetuned_model.onnx");
        log::info!("Loading finetuned_model from {:?}", path);

        if !path.exists() {
            return Err(anyhow!("Model file not found: {:?}", path));
        }

        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(&path)
            .context("Failed to load finetuned_model ONNX model")?;

        Ok(Phase2FinetunedModel {
            session: std::sync::Arc::new(session),
            model_path: path,
        })
    }
}

/// CSS 选择器嵌入模型推理器
#[cfg(feature = "onnx")]
pub struct Phase2SelectorEmbedding {
    session: std::sync::Arc<Session>,
    model_path: std::path::PathBuf,
}

#[cfg(feature = "onnx")]
impl Phase2SelectorEmbedding {
    /// 执行选择器嵌入推理
    ///
    /// # 参数
    /// - `input_tokens`: [batch_size, seq_len] 的标记化选择器
    ///
    /// # 返回
    /// 128 维嵌入向量，形状为 [batch_size, seq_len, 128]
    pub fn infer(&self, input_tokens: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        let batch_size = input_tokens.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let seq_len = input_tokens[0].len();

        // 将输入转换为 ONNX 张量
        let mut flat_tokens = Vec::with_capacity(batch_size * seq_len);
        for tokens in input_tokens {
            if tokens.len() != seq_len {
                return Err(anyhow!(
                    "All input sequences must have the same length. Expected {}, got {}",
                    seq_len,
                    tokens.len()
                ));
            }
            flat_tokens.extend(tokens);
        }

        let input_tensor = OrtTensor::from_array(([batch_size, seq_len], flat_tokens.into()))?;
        let outputs = self.session.run(vec![input_tensor])?;

        if outputs.is_empty() {
            return Err(anyhow!("No output from selector_embedding model"));
        }

        // 提取输出并转换为 Vec<Vec<f32>>
        let output = &outputs[0];
        let data: &[f32] = output.try_extract_tensor()?.view().as_slice();

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * seq_len * 128;
            let end = start + seq_len * 128;
            if end <= data.len() {
                results.push(data[start..end].to_vec());
            }
        }

        Ok(results)
    }

    /// 获取模型路径（用于调试）
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }
}

/// CSS 属性预测模型推理器
#[cfg(feature = "onnx")]
pub struct Phase2PropertyPredictor {
    session: std::sync::Arc<Session>,
    model_path: std::path::PathBuf,
}

#[cfg(feature = "onnx")]
impl Phase2PropertyPredictor {
    /// 执行属性预测推理
    ///
    /// # 参数
    /// - `embeddings`: 选择器嵌入，形状为 [batch_size, 10, 128]
    ///
    /// # 返回
    /// 50 个 CSS 属性的概率，形状为 [batch_size, 50]，值在 [0, 1]
    pub fn infer(&self, embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let batch_size = embeddings.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let embedding_dim = 128 * 10; // 假设输入是 [batch, 10, 128]

        // 验证输入维度
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != embedding_dim {
                log::warn!(
                    "Embedding {} has length {}, expected {}",
                    i,
                    emb.len(),
                    embedding_dim
                );
            }
        }

        // 将嵌入展平为单个张量
        let mut flat_embeddings = Vec::with_capacity(batch_size * embedding_dim);
        for emb in embeddings {
            flat_embeddings.extend(emb);
        }

        let input_tensor = OrtTensor::from_array(([batch_size, 10, 128], flat_embeddings.into()))?;
        let outputs = self.session.run(vec![input_tensor])?;

        if outputs.is_empty() {
            return Err(anyhow!("No output from property_predictor model"));
        }

        // 提取输出
        let output = &outputs[0];
        let data: &[f32] = output.try_extract_tensor()?.view().as_slice();

        let mut results = Vec::with_capacity(batch_size);
        let prop_dim = 50; // 50 个 CSS 属性
        for i in 0..batch_size {
            let start = i * prop_dim;
            let end = start + prop_dim;
            if end <= data.len() {
                results.push(data[start..end].to_vec());
            }
        }

        Ok(results)
    }

    /// 获取模型路径
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }
}

/// 颜色学习模型推理器
#[cfg(feature = "onnx")]
pub struct Phase2ColorModel {
    session: std::sync::Arc<Session>,
    model_path: std::path::PathBuf,
}

#[cfg(feature = "onnx")]
impl Phase2ColorModel {
    /// 执行颜色特征提取
    ///
    /// # 参数
    /// - `rgb_images`: RGB 图像，形状为 [batch_size, 3, 32, 32]
    ///
    /// # 返回
    /// 颜色特征向量
    pub fn infer(&self, rgb_images: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let batch_size = rgb_images.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let image_size = 3 * 32 * 32; // [3, 32, 32]

        // 合并批次输入
        let mut flat_images = Vec::with_capacity(batch_size * image_size);
        for img in rgb_images {
            if img.len() != image_size {
                log::warn!("Image has size {}, expected {}", img.len(), image_size);
            }
            flat_images.extend(img);
        }

        let input_tensor = OrtTensor::from_array(([batch_size, 3, 32, 32], flat_images.into()))?;
        let outputs = self.session.run(vec![input_tensor])?;

        if outputs.is_empty() {
            return Err(anyhow!("No output from color_model"));
        }

        // 提取输出
        let output = &outputs[0];
        let data: &[f32] = output.try_extract_tensor()?.view().as_slice();

        // 推断输出维度
        let output_size = data.len() / batch_size;
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * output_size;
            let end = start + output_size;
            if end <= data.len() {
                results.push(data[start..end].to_vec());
            }
        }

        Ok(results)
    }

    /// 获取模型路径
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }
}

/// 完整页面模型推理器
#[cfg(feature = "onnx")]
pub struct Phase2CompleteModel {
    session: std::sync::Arc<Session>,
    model_path: std::path::PathBuf,
}

#[cfg(feature = "onnx")]
impl Phase2CompleteModel {
    /// 执行完整页面分析推理
    pub fn infer(&self, features: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let batch_size = features.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let feature_dim = features[0].len();

        // 合并批次输入
        let mut flat_features = Vec::with_capacity(batch_size * feature_dim);
        for feat in features {
            if feat.len() != feature_dim {
                return Err(anyhow!(
                    "All features must have the same dimension. Expected {}, got {}",
                    feature_dim,
                    feat.len()
                ));
            }
            flat_features.extend(feat);
        }

        let input_tensor =
            OrtTensor::from_array(([batch_size, feature_dim], flat_features.into()))?;
        let outputs = self.session.run(vec![input_tensor])?;

        if outputs.is_empty() {
            return Err(anyhow!("No output from complete_model"));
        }

        let output = &outputs[0];
        let data: &[f32] = output.try_extract_tensor()?.view().as_slice();

        let output_size = data.len() / batch_size;
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * output_size;
            let end = start + output_size;
            if end <= data.len() {
                results.push(data[start..end].to_vec());
            }
        }

        Ok(results)
    }

    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }
}

/// 微调模型推理器（LoRA 适配器）
#[cfg(feature = "onnx")]
pub struct Phase2FinetunedModel {
    session: std::sync::Arc<Session>,
    model_path: std::path::PathBuf,
}

#[cfg(feature = "onnx")]
impl Phase2FinetunedModel {
    /// 执行微调推理
    pub fn infer(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let batch_size = inputs.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        let input_dim = inputs[0].len();

        let mut flat_inputs = Vec::with_capacity(batch_size * input_dim);
        for inp in inputs {
            if inp.len() != input_dim {
                return Err(anyhow!(
                    "All inputs must have the same dimension. Expected {}, got {}",
                    input_dim,
                    inp.len()
                ));
            }
            flat_inputs.extend(inp);
        }

        let input_tensor = OrtTensor::from_array(([batch_size, input_dim], flat_inputs.into()))?;
        let outputs = self.session.run(vec![input_tensor])?;

        if outputs.is_empty() {
            return Err(anyhow!("No output from finetuned_model"));
        }

        let output = &outputs[0];
        let data: &[f32] = output.try_extract_tensor()?.view().as_slice();

        let output_size = data.len() / batch_size;
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * output_size;
            let end = start + output_size;
            if end <= data.len() {
                results.push(data[start..end].to_vec());
            }
        }

        Ok(results)
    }

    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase2_loader_creation() {
        let loader = Phase2ModelLoader::new("models");
        assert_eq!(loader.model_dir.to_str().unwrap(), "models");
    }
}
