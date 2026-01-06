#![cfg(feature = "ai")]
// AI网站再生成模块
// 使用ONNX模型将原始网站代码转换为简化版本

use anyhow::{Context, Result};
use ort::{Session, Value};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// 网站再生成器
/// 输入：原始HTML+CSS+JS
/// 输出：AI简化版HTML+CSS+JS
pub struct WebsiteRegenerator {
    session: Arc<Session>,
    char2idx: HashMap<char, i64>,
    idx2char: HashMap<i64, char>,
    vocab_size: usize,
    max_len: usize,
}

impl WebsiteRegenerator {
    /// 从ONNX模型创建再生成器
    pub fn new<P: AsRef<Path>>(model_path: P, config_path: P) -> Result<Self> {
        log::info!("Loading website regeneration model...");

        // 加载ONNX模型
        let session = Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        log::info!("Model loaded successfully");

        // 加载配置（包括vocab映射）
        let config_str =
            std::fs::read_to_string(config_path).context("Failed to read config file")?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;

        let vocab_size = config["vocab_size"].as_u64().unwrap_or(229) as usize;
        let max_len = config["max_len"].as_u64().unwrap_or(2048) as usize;

        // TODO: 从checkpoint加载实际的char2idx映射
        // 这里使用简化版本，实际应从训练checkpoind获取
        let char2idx = Self::build_default_vocab();
        let idx2char = char2idx.iter().map(|(k, v)| (*v, *k)).collect();

        Ok(Self {
            session: Arc::new(session),
            char2idx,
            idx2char,
            vocab_size,
            max_len,
        })
    }

    /// 构建默认词汇表（简化版）
    fn build_default_vocab() -> HashMap<char, i64> {
        let mut vocab = HashMap::new();
        vocab.insert('\0', 0); // PAD
        vocab.insert('\x01', 1); // SOS
        vocab.insert('\x02', 2); // EOS

        // 添加常见字符
        let common_chars = " \n\t<>/=\"'{}[];():.#,-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        for (i, ch) in common_chars.chars().enumerate() {
            vocab.insert(ch, (i + 3) as i64);
        }

        vocab
    }

    /// 编码文本为token IDs
    fn encode(&self, text: &str) -> Vec<i64> {
        let mut tokens = vec![1]; // SOS

        for ch in text.chars().take(self.max_len - 2) {
            let token = self.char2idx.get(&ch).copied().unwrap_or(0);
            tokens.push(token);
        }

        tokens.push(2); // EOS

        // Padding到max_len
        while tokens.len() < self.max_len {
            tokens.push(0); // PAD
        }

        tokens
    }

    /// 解码token IDs为文本
    fn decode(&self, tokens: &[i64]) -> String {
        tokens
            .iter()
            .filter_map(|&t| {
                if t == 0 || t == 1 || t == 2 {
                    None // 跳过特殊token
                } else {
                    self.idx2char.get(&t).copied()
                }
            })
            .collect()
    }

    /// 再生成网站代码
    ///
    /// # Arguments
    /// * `original_code` - 原始网站代码（HTML+CSS+JS合并）
    ///
    /// # Returns
    /// AI简化版代码
    pub fn regenerate(&self, original_code: &str) -> Result<String> {
        log::debug!(
            "Regenerating website code (length: {})",
            original_code.len()
        );

        // 编码输入
        let src_tokens = self.encode(original_code);
        let src_len = src_tokens.len();

        // 创建初始target（只有SOS）
        let mut tgt_tokens = vec![1i64]; // SOS

        // 自回归生成
        for _ in 0..self.max_len {
            let tgt_len = tgt_tokens.len();

            // Padding target
            let mut tgt_padded = tgt_tokens.clone();
            while tgt_padded.len() < self.max_len {
                tgt_padded.push(0);
            }

            // 创建ONNX输入
            let src_array = ndarray::Array2::from_shape_vec((1, src_len), src_tokens.clone())?;
            let tgt_array = ndarray::Array2::from_shape_vec((1, self.max_len), tgt_padded)?;

            let src_value = Value::from_array(src_array)?;
            let tgt_value = Value::from_array(tgt_array)?;

            // 推理
            let outputs = self.session.run(ort::inputs![
                "src" => src_value,
                "tgt" => tgt_value,
            ]?)?;

            // 获取输出logits: [1, tgt_len, vocab_size]
            let logits = outputs[0].try_extract_tensor::<f32>()?.view().to_owned();

            // 取最后一个位置的logits
            let last_logits = logits.slice(ndarray::s![0, tgt_len - 1, ..]).to_owned();

            // Greedy decoding: 选择概率最高的token
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as i64)
                .unwrap_or(0);

            // 如果生成了EOS，停止
            if next_token == 2 {
                break;
            }

            tgt_tokens.push(next_token);
        }

        // 解码输出
        let regenerated = self.decode(&tgt_tokens);

        log::debug!(
            "Regeneration complete (output length: {})",
            regenerated.len()
        );

        Ok(regenerated)
    }

    /// 从完整网站HTML中再生成
    /// 自动提取HTML/CSS/JS并合并
    pub fn regenerate_from_html(&self, html: &str) -> Result<RegeneratedWebsite> {
        // 提取CSS和JS
        let css = Self::extract_css_from_html(html);
        let js = Self::extract_js_from_html(html);

        // 合并代码
        let combined = format!("{}\n{}\n{}", html, css, js);

        // 再生成
        let regenerated = self.regenerate(&combined)?;

        // 分割回HTML/CSS/JS（简单分割，实际可能需要更复杂的解析）
        let parts: Vec<&str> = regenerated.split('\n').collect();
        let html_part = parts.get(0).copied().unwrap_or("");
        let css_part = parts.get(1).copied().unwrap_or("");
        let js_part = parts.get(2).copied().unwrap_or("");

        Ok(RegeneratedWebsite {
            html: html_part.to_string(),
            css: css_part.to_string(),
            js: js_part.to_string(),
        })
    }

    fn extract_css_from_html(html: &str) -> String {
        // 简单提取<style>标签内容
        // TODO: 使用html5ever解析
        let mut css = String::new();
        if let Some(start) = html.find("<style>") {
            if let Some(end) = html[start..].find("</style>") {
                css = html[start + 7..start + end].to_string();
            }
        }
        css
    }

    fn extract_js_from_html(html: &str) -> String {
        // 简单提取<script>标签内容
        // TODO: 使用html5ever解析
        let mut js = String::new();
        if let Some(start) = html.find("<script>") {
            if let Some(end) = html[start..].find("</script>") {
                js = html[start + 8..start + end].to_string();
            }
        }
        js
    }
}

/// 再生成的网站
#[derive(Debug, Clone)]
pub struct RegeneratedWebsite {
    pub html: String,
    pub css: String,
    pub js: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // 需要模型文件
    fn test_regeneration() {
        let regenerator = WebsiteRegenerator::new(
            "../../models/local/website_generator_v1.onnx",
            "../../models/local/website_generator_v1_config.json",
        )
        .unwrap();

        let original = "<html><head><style>.container{width:100%}</style></head><body><div class=\"container\">Hello</div></body></html>";

        let result = regenerator.regenerate(original).unwrap();

        println!("Original: {}", original);
        println!("Regenerated: {}", result);

        assert!(!result.is_empty());
    }
}
