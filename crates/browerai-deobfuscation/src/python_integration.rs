// Python反混淆系统集成模块
//
// 集成全球JS混淆/反混淆知识库和深度学习模型

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;

/// Python反混淆结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonDeobfuscationResult {
    /// 原始代码
    pub original: String,
    /// 反混淆后的代码
    pub deobfuscated: String,
    /// 检测到的混淆器
    pub detected_obfuscators: Vec<(String, f64)>, // (name, confidence)
    /// 应用的规则
    pub applied_rules: Vec<String>,
    /// 代码长度减少比例
    pub reduction_ratio: f64,
    /// 是否成功
    pub success: bool,
}

/// 混淆器信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObfuscatorInfo {
    pub name: String,
    pub country: String,
    pub difficulty: String, // Low, Medium, High, Extreme
    pub techniques: Vec<String>,
}

/// Python反混淆系统接口
pub struct PythonDeobfuscationSystem {
    /// Python解释器路径
    python_path: String,
    /// 训练目录路径
    training_dir: PathBuf,
}

impl PythonDeobfuscationSystem {
    /// 创建新的Python反混淆系统
    pub fn new() -> Result<Self> {
        let training_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .context("Failed to get parent directory")?
            .parent()
            .context("Failed to get project root")?
            .join("training");

        Ok(Self {
            python_path: "python3".to_string(),
            training_dir,
        })
    }

    /// 设置Python解释器路径
    pub fn with_python_path(mut self, path: impl Into<String>) -> Self {
        self.python_path = path.into();
        self
    }

    /// 反混淆JS代码
    pub fn deobfuscate(&self, code: &str) -> Result<PythonDeobfuscationResult> {
        // 创建临时Python脚本
        let script = format!(
            r#"
import sys
import json
sys.path.insert(0, r'{}')

from global_js_obfuscation_deobfuscation_system import PracticalDeobfuscator

deobfuscator = PracticalDeobfuscator()
code = {}
result = deobfuscator.deobfuscate(code)

# 输出JSON
output = {{
    'original': result['original'],
    'deobfuscated': result['deobfuscated'],
    'detected_obfuscators': result['improvement']['detected_obfuscators'],
    'applied_rules': result['improvement']['applied_rules'],
    'reduction_ratio': result['improvement']['reduction_ratio'],
    'success': result['success'],
}}

print(json.dumps(output, ensure_ascii=False))
"#,
            self.training_dir.display(),
            serde_json::to_string(code)?
        );

        // 执行Python脚本
        let output = Command::new(&self.python_path)
            .arg("-c")
            .arg(&script)
            .output()
            .context("Failed to execute Python")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Python script failed: {}", stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: PythonDeobfuscationResult =
            serde_json::from_str(&stdout).context("Failed to parse Python output")?;

        Ok(result)
    }

    /// 检测混淆器类型
    pub fn detect_obfuscator(&self, code: &str) -> Result<Vec<ObfuscatorInfo>> {
        let script = format!(
            r#"
import sys
import json
sys.path.insert(0, r'{}')

from global_js_obfuscation_deobfuscation_system import PracticalDeobfuscator

deobfuscator = PracticalDeobfuscator()
code = {}
detected = deobfuscator.detect_obfuscator(code)

output = [
    {{
        'name': obf.name,
        'country': obf.country,
        'difficulty': obf.difficulty,
        'techniques': [t.value for t in obf.techniques],
    }}
    for obf, score in detected[:5]
]

print(json.dumps(output, ensure_ascii=False))
"#,
            self.training_dir.display(),
            serde_json::to_string(code)?
        );

        let output = Command::new(&self.python_path)
            .arg("-c")
            .arg(&script)
            .output()
            .context("Failed to execute Python")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Python script failed: {}", stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: Vec<ObfuscatorInfo> =
            serde_json::from_str(&stdout).context("Failed to parse Python output")?;

        Ok(result)
    }

    /// 获取知识库统计信息
    pub fn get_statistics(&self) -> Result<KnowledgeBaseStatistics> {
        let script = format!(
            r#"
import sys
import json
sys.path.insert(0, r'{}')

from global_js_obfuscation_deobfuscation_system import GlobalObfuscationKnowledgeBase

kb = GlobalObfuscationKnowledgeBase()
stats = kb.get_statistics()

print(json.dumps(stats, ensure_ascii=False))
"#,
            self.training_dir.display()
        );

        let output = Command::new(&self.python_path)
            .arg("-c")
            .arg(&script)
            .output()
            .context("Failed to execute Python")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Python script failed: {}", stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: KnowledgeBaseStatistics =
            serde_json::from_str(&stdout).context("Failed to parse Python output")?;

        Ok(result)
    }
}

impl Default for PythonDeobfuscationSystem {
    fn default() -> Self {
        Self::new().expect("Failed to create PythonDeobfuscationSystem")
    }
}

/// 知识库统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseStatistics {
    pub total: usize,
    pub open_source: usize,
    pub commercial: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_deobfuscate_hex_encoding() {
        let system = PythonDeobfuscationSystem::new().unwrap();

        let code = r"var msg = '\x48\x65\x6c\x6c\x6f';";
        let result = system.deobfuscate(code).unwrap();

        println!("Original: {}", result.original);
        println!("Deobfuscated: {}", result.deobfuscated);
        println!("Detected: {:?}", result.detected_obfuscators);
        println!("Rules: {:?}", result.applied_rules);
    }

    #[test]
    #[ignore]
    fn test_detect_obfuscator() {
        let system = PythonDeobfuscationSystem::new().unwrap();

        let code = r"var _0x1234 = '\x48\x65\x6c\x6c\x6f';";
        let obfuscators = system.detect_obfuscator(code).unwrap();

        println!("Detected obfuscators:");
        for obf in obfuscators {
            println!("  - {} ({}) [{}]", obf.name, obf.country, obf.difficulty);
        }
    }

    #[test]
    #[ignore]
    fn test_get_statistics() {
        let system = PythonDeobfuscationSystem::new().unwrap();

        let stats = system.get_statistics().unwrap();

        println!("Knowledge Base Statistics:");
        println!("  Total: {}", stats.total);
        println!("  Open Source: {}", stats.open_source);
        println!("  Commercial: {}", stats.commercial);
    }
}
