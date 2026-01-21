//! ONNX 推理实现 - Phase B
//!
//! 实现真实的 ONNX 模型推理，替换 TODO 占位符

use anyhow::{Context, Result};
use std::collections::HashMap;

#[cfg(feature = "ai")]
use ort::value::Value;

/// ONNX 推理辅助函数
pub struct OnnxInference {
    vocab: HashMap<String, i64>,
    reverse_vocab: HashMap<i64, String>,
}

impl Default for OnnxInference {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxInference {
    /// 创建新的推理辅助器
    pub fn new() -> Self {
        let vocab = Self::build_vocab();
        let reverse_vocab = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        Self {
            vocab,
            reverse_vocab,
        }
    }

    /// 构建词汇表
    fn build_vocab() -> HashMap<String, i64> {
        let mut vocab = HashMap::new();

        // 特殊 tokens
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("<unk>".to_string(), 1);
        vocab.insert("<start>".to_string(), 2);
        vocab.insert("<end>".to_string(), 3);

        // 常见 JS tokens
        let common_tokens = vec![
            "function", "var", "let", "const", "return", "if", "else", "for", "while", "forEach",
            "map", "filter", "reduce", "push", "pop", "shift", "unshift", "length", "indexOf",
            "slice", "splice", "join", "split", "replace", "promise", "async", "await", "then",
            "catch", "finally", "data", "result", "value", "item", "element", "index", "key",
            "handler", "callback", "event", "listener", "error",
        ];

        for (idx, token) in common_tokens.iter().enumerate() {
            vocab.insert(token.to_string(), (idx + 4) as i64);
        }

        vocab
    }

    /// 将代码 tokenize
    pub fn tokenize(&self, code: &str) -> Vec<i64> {
        // 简单的基于空格和符号的分词
        let tokens: Vec<&str> = code
            .split(|c: char| c.is_whitespace() || "{}()[];,".contains(c))
            .filter(|s| !s.is_empty())
            .collect();

        tokens
            .iter()
            .map(|token| {
                *self.vocab.get(*token).unwrap_or(&1) // <unk> token
            })
            .collect()
    }

    /// 将 token IDs 转换回文本
    pub fn detokenize(&self, token_ids: &[i64]) -> String {
        token_ids
            .iter()
            .filter_map(|id| self.reverse_vocab.get(id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// 获取词汇表的引用
    pub fn vocab(&self) -> &HashMap<String, i64> {
        &self.vocab
    }

    /// 检查词汇表中是否存在某个 token
    pub fn has_token(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    /// 执行 ONNX 推理（仅在 ai 特性启用时）
    #[cfg(feature = "ai")]
    pub fn infer_with_session(
        &self,
        session: &mut ort::session::Session,
        code: &str,
        max_length: usize,
    ) -> Result<String> {
        use ort::session::input::SessionInputValue;

        // 1. Tokenize 输入代码
        let mut token_ids = self.tokenize(code);

        // 2. Padding/Truncation 到固定长度
        token_ids.resize(max_length, 0);

        // 3. 转换为 f32 用于 ONNX 模型输入
        let input_data: Vec<f32> = token_ids.iter().map(|&x| x as f32).collect();

        // 4. 准备输入形状 (batch_size=1, seq_len=max_length)
        let shape = vec![1i64, max_length as i64];

        // 5. 使用 ort 2.0 API 创建输入张量
        let input_tensor =
            Value::from_array((shape, input_data)).context("Failed to create input tensor")?;

        // 6. 运行推理
        let inputs = [SessionInputValue::from(input_tensor)];
        let outputs = session.run(inputs).context("ONNX inference failed")?;

        // 7. 解析输出
        if outputs.len() == 0 {
            return Err(anyhow::anyhow!("No outputs from ONNX inference"));
        }

        // 8. 提取输出数据（通常是 f32）
        let output_f32 = outputs[0]
            .try_extract_array::<f32>()
            .context("Failed to extract output as float32")?;

        // 9. 转换回 token IDs（取 argmax 或简单转换）
        let predicted_ids: Vec<i64> = output_f32
            .iter()
            .take(max_length)
            .map(|&x| x as i64)
            .collect();

        // 10. Detokenize 回文本
        Ok(self.detokenize(&predicted_ids))
    }

    /// 批量推理（仅在 ai 特性启用时）
    #[cfg(feature = "ai")]
    pub fn batch_infer(
        &self,
        session: &mut ort::session::Session,
        codes: &[String],
        max_length: usize,
    ) -> Result<Vec<String>> {
        codes
            .iter()
            .map(|code| self.infer_with_session(session, code, max_length))
            .collect()
    }

    /// 执行推理的 stub 版本（当 ai 特性未启用时）
    #[cfg(not(feature = "ai"))]
    pub fn infer_with_session(
        &self,
        _session: &mut std::any::Any,
        code: &str,
        _max_length: usize,
    ) -> Result<String> {
        log::warn!("ONNX inference not available (ai feature not enabled)");
        // 返回原始代码作为回退
        Ok(code.to_string())
    }

    /// 批量推理的 stub 版本
    #[cfg(not(feature = "ai"))]
    pub fn batch_infer(
        &self,
        _session: &mut std::any::Any,
        codes: &[String],
        _max_length: usize,
    ) -> Result<Vec<String>> {
        log::warn!("ONNX batch inference not available (ai feature not enabled)");
        Ok(codes.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let helper = OnnxInference::new();
        let tokens = helper.tokenize("function foo() { return 42; }");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_vocab_contains_common_tokens() {
        let helper = OnnxInference::new();
        assert!(helper.vocab.contains_key("function"));
        assert!(helper.vocab.contains_key("return"));
        assert!(helper.vocab.contains_key("data"));
    }

    #[test]
    fn test_detokenize() {
        let helper = OnnxInference::new();
        let token_ids = vec![4, 5, 6]; // function var let
        let text = helper.detokenize(&token_ids);
        assert!(!text.is_empty());
    }

    #[test]
    fn test_reverse_vocab_consistency() {
        let helper = OnnxInference::new();
        // 验证 reverse vocab 与 vocab 一致
        for (token, id) in &helper.vocab {
            assert_eq!(helper.reverse_vocab.get(id), Some(&token.clone()));
        }
    }
}
