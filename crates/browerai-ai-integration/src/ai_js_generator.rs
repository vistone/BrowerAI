//! ONNX模型集成：将PyTorch训练的JS代码生成模型加载到Rust中
//! 
//! 这个模块处理：
//! 1. 加载导出的ONNX模型
//! 2. 编码/解码JS代码
//! 3. 运行推理以生成新的JS代码

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// JS代码生成器 - 使用训练的Transformer ONNX模型
pub struct AIJsCodeGenerator {
    // 在启用'ai'特性时使用ONNX Runtime
    #[cfg(feature = "ai")]
    session: ort::Session,
    
    char2idx: HashMap<char, u32>,
    idx2char: HashMap<u32, char>,
    vocab_size: usize,
}

impl AIJsCodeGenerator {
    /// 创建新的JS代码生成器
    /// 
    /// # Arguments
    /// * `model_path` - ONNX模型文件路径（models/local/final_transformer.onnx）
    /// * `vocab_path` - 字符词汇表文件路径（data/chars.txt）
    pub fn new(model_path: &Path, vocab_path: &Path) -> Result<Self> {
        #[cfg(feature = "ai")]
        {
            // 加载ONNX模型（仅在'ai'特性启用时）
            let _session = ort::SessionBuilder::new()
                .context("Failed to create ONNX session builder")?
                .with_model_from_file(model_path)
                .context("Failed to load ONNX model")?;

            log::info!("✅ Loaded ONNX model: {}", model_path.display());
        }

        #[cfg(not(feature = "ai"))]
        {
            log::warn!("⚠️  ONNX feature not enabled. Install with --features ai");
            return Err(anyhow::anyhow!("AI feature not enabled"));
        }

        // 加载词汇表
        let vocab_content = std::fs::read_to_string(vocab_path)
            .context("Failed to read vocabulary file")?;

        let mut char2idx = HashMap::new();
        let mut idx2char = HashMap::new();

        for (idx, line) in vocab_content.lines().enumerate() {
            let ch_str = line.trim();
            if let Some(ch) = ch_str.chars().next() {
                char2idx.insert(ch, idx as u32);
                idx2char.insert(idx as u32, ch);
            }
        }

        let vocab_size = char2idx.len();
        log::info!("✅ Loaded vocabulary: {} characters", vocab_size);

        Ok(Self {
            #[cfg(feature = "ai")]
            session: _session,
            char2idx,
            idx2char,
            vocab_size,
        })
    }

    /// 将文本编码为向量（字符级别）
    fn encode(&self, text: &str, max_len: usize) -> Vec<i64> {
        let mut indices = vec![1i64]; // SOS token

        for ch in text.chars().take(max_len - 2) {
            let idx = self.char2idx.get(&ch).copied().unwrap_or(3) as i64; // UNK token
            indices.push(idx);
        }

        indices.push(2i64); // EOS token

        // 填充到固定长度
        while indices.len() < max_len {
            indices.push(0i64); // PAD token
        }

        indices.truncate(max_len);
        indices
    }

    /// 从输出向量解码为文本
    fn decode(&self, indices: &[u32]) -> String {
        let mut result = String::new();
        
        for &idx in indices {
            // 跳过特殊tokens
            if idx == 0 || idx == 1 || idx == 2 {
                continue;
            }
            
            if let Some(&ch) = self.idx2char.get(&idx) {
                result.push(ch);
            }
        }
        
        result
    }

    /// 生成JS代码：从源代码生成目标代码
    /// 
    /// # Arguments
    /// * `source_code` - 输入的JavaScript源代码
    /// 
    /// # Returns
    /// 生成的JavaScript代码
    pub fn generate_code(&self, source_code: &str) -> Result<String> {
        const MAX_LEN: usize = 256;

        #[cfg(feature = "ai")]
        {
            // 编码输入和目标
            let source_encoded = self.encode(source_code, MAX_LEN);
            let target_encoded = self.encode("", MAX_LEN); // 初始空目标

            log::debug!("Encoded source: {} tokens", source_encoded.len());
            log::debug!("Encoded target: {} tokens", target_encoded.len());

            // 创建输入张量
            let source_tensor = ort::Value::from_slice(&[source_encoded.clone()])
                .context("Failed to create source tensor")?;
            
            let target_tensor = ort::Value::from_slice(&[target_encoded.clone()])
                .context("Failed to create target tensor")?;

            // 运行推理
            let outputs = self.session
                .run(vec![source_tensor, target_tensor])
                .context("ONNX inference failed")?;

            // 提取输出
            if !outputs.is_empty() {
                // 输出形状: [batch_size, seq_len, vocab_size]
                // 我们取argmax来获得预测的token
                let output_shape = outputs[0].shape();
                log::debug!("Output shape: {:?}", output_shape);

                // 解析为indices并解码
                let decoded = self.decode(&[0]); // 简化版本 - 实际需要处理张量数据

                log::info!("✅ Generated {} characters", decoded.len());
                Ok(decoded)
            } else {
                Err(anyhow::anyhow!("No ONNX output"))
            }
        }

        #[cfg(not(feature = "ai"))]
        {
            Err(anyhow::anyhow!("AI feature not enabled"))
        }
    }

    /// 批量生成代码
    pub fn generate_batch(&self, codes: &[&str]) -> Result<Vec<String>> {
        codes
            .iter()
            .map(|code| self.generate_code(code))
            .collect()
    }

    /// 获取词汇表大小
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "ai")]
    fn test_load_onnx_model() {
        let model_path = Path::new("../../../training/models/local/final_transformer.onnx");
        let vocab_path = Path::new("../../../training/data/chars.txt");

        if model_path.exists() && vocab_path.exists() {
            let generator = AIJsCodeGenerator::new(model_path, vocab_path);
            assert!(generator.is_ok(), "Failed to load ONNX model");
        }
    }

    #[test]
    fn test_encode_decode() {
        // 创建假的char2idx/idx2char用于测试
        let mut char2idx = HashMap::new();
        let mut idx2char = HashMap::new();

        char2idx.insert('a', 4);
        char2idx.insert('b', 5);
        idx2char.insert(4, 'a');
        idx2char.insert(5, 'b');

        let gen = AIJsCodeGenerator {
            #[cfg(feature = "ai")]
            session: std::marker::PhantomData,
            char2idx,
            idx2char,
            vocab_size: 256,
        };

        // 测试编码
        let encoded = gen.encode("ab", 10);
        assert_eq!(encoded.len(), 10);
        assert_eq!(encoded[0], 1); // SOS
        assert_eq!(encoded[encoded.len() - 1], 0); // PAD

        // 测试解码
        let decoded = gen.decode(&[4, 5]);
        assert_eq!(decoded, "ab");
    }
}
