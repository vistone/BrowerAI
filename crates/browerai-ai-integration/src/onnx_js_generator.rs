// Rust集成：加载ONNX模型进行JS代码生成
use ort::{Session, SessionBuilder, Value};
use std::path::Path;

pub struct JsCodeGenerator {
    session: Session,
    char2idx: std::collections::HashMap<char, u32>,
    idx2char: std::collections::HashMap<u32, char>,
}

impl JsCodeGenerator {
    /// 从ONNX模型加载JS代码生成器
    pub fn new(model_path: &Path, vocab_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // 加载ONNX模型
        let session = SessionBuilder::new()?
            .with_model_from_file(model_path)?;

        // 加载字符词汇表
        let vocab_content = std::fs::read_to_string(vocab_path)?;
        let mut char2idx = std::collections::HashMap::new();
        let mut idx2char = std::collections::HashMap::new();

        for (idx, line) in vocab_content.lines().enumerate() {
            let char = line.trim();
            if !char.is_empty() {
                char2idx.insert(char.chars().next().unwrap(), idx as u32);
                idx2char.insert(idx as u32, char.chars().next().unwrap());
            }
        }

        Ok(Self {
            session,
            char2idx,
            idx2char,
        })
    }

    /// 编码输入源代码为向量
    fn encode(&self, text: &str, max_len: usize) -> Vec<i64> {
        let mut indices = vec![1i64]; // SOS
        
        for ch in text.chars().take(max_len - 2) {
            let idx = self.char2idx.get(&ch).copied().unwrap_or(3) as i64; // UNK
            indices.push(idx);
        }
        
        indices.push(2i64); // EOS
        
        while indices.len() < max_len {
            indices.push(0i64); // PAD
        }
        
        indices.truncate(max_len);
        indices
    }

    /// 解码输出向量为JS代码
    fn decode(&self, indices: Vec<u32>) -> String {
        let mut result = String::new();
        for idx in indices {
            if idx == 0 || idx == 1 || idx == 2 {
                continue; // Skip PAD, SOS, EOS
            }
            if let Some(ch) = self.idx2char.get(&idx) {
                result.push(*ch);
            }
        }
        result
    }

    /// 生成JS代码：输入源代码 → 输出目标JS
    pub fn generate(&self, source_code: &str) -> Result<String, Box<dyn std::error::Error>> {
        const MAX_LEN: usize = 256;
        
        // 编码输入
        let source_encoded = self.encode(source_code, MAX_LEN);
        let target_encoded = self.encode("", MAX_LEN); // 初始空目标
        
        // 创建输入张量
        let source_tensor = Value::from_slice(&[source_encoded])
            .map_err(|e| format!("Failed to create source tensor: {}", e))?;
        let target_tensor = Value::from_slice(&[target_encoded])
            .map_err(|e| format!("Failed to create target tensor: {}", e))?;
        
        // 运行推理
        let outputs = self.session.run(vec![source_tensor, target_tensor])
            .map_err(|e| format!("ONNX inference failed: {}", e))?;
        
        // 解析输出并解码
        if !outputs.is_empty() {
            // 这里需要根据实际的ONNX输出形状来处理
            let output_shape = outputs[0].shape();
            let output_data = outputs[0].try_extract_raw_tensor::<i64>()?;
            
            let mut result_indices = Vec::new();
            for val in output_data.iter().take(MAX_LEN) {
                result_indices.push(*val as u32);
            }
            
            Ok(self.decode(result_indices))
        } else {
            Err("No ONNX output".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_onnx_model() {
        let model_path = Path::new("models/local/final_transformer.onnx");
        let vocab_path = Path::new("data/chars.txt");
        
        if model_path.exists() && vocab_path.exists() {
            let generator = JsCodeGenerator::new(model_path, vocab_path);
            assert!(generator.is_ok());
        }
    }

    #[test]
    fn test_encode_decode() {
        let model_path = Path::new("models/local/final_transformer.onnx");
        let vocab_path = Path::new("data/chars.txt");
        
        if model_path.exists() && vocab_path.exists() {
            if let Ok(gen) = JsCodeGenerator::new(model_path, vocab_path) {
                let text = "const x = 42;";
                let encoded = gen.encode(text, 256);
                
                assert_eq!(encoded[0], 1); // SOS
                assert_eq!(encoded[encoded.len()-1], 0); // PAD
            }
        }
    }
}
