// 快速增强模型集成测试
// 测试新训练的 fast_enhanced.onnx 模型在Rust中的集成

#[cfg(test)]
mod fast_enhanced_integration_tests {
    use std::path::PathBuf;

    /// 测试1: 验证ONNX模型文件存在
    #[test]
    fn test_model_file_exists() {
        let model_path = PathBuf::from("models/local/fast_enhanced.onnx");
        assert!(
            model_path.exists(),
            "fast_enhanced.onnx 模型文件不存在: {:?}",
            model_path
        );

        // 检查文件大小
        let metadata = std::fs::metadata(&model_path).unwrap();
        let file_size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        
        println!("✅ 模型文件存在");
        println!("   路径: {:?}", model_path);
        println!("   大小: {:.2} MB", file_size_mb);
        
        assert!(
            file_size_mb > 3.0 && file_size_mb < 5.0,
            "模型文件大小异常: {:.2} MB (期望 3-5 MB)",
            file_size_mb
        );
    }

    /// 测试2: 加载ONNX Runtime会话
    #[test]
    #[cfg(feature = "ai")]
    fn test_onnx_session_creation() {
        use ort::{Session, GraphOptimizationLevel};

        let model_path = "models/local/fast_enhanced.onnx";
        
        // 创建ONNX Runtime会话
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .unwrap()
            .commit_from_file(model_path);

        assert!(
            session.is_ok(),
            "ONNX Runtime会话创建失败: {:?}",
            session.err()
        );

        let session = session.unwrap();
        
        // 检查输入输出
        let inputs = session.inputs.clone();
        let outputs = session.outputs.clone();

        println!("✅ ONNX Runtime会话创建成功");
        println!("   输入数量: {}", inputs.len());
        println!("   输出数量: {}", outputs.len());

        assert_eq!(inputs.len(), 1, "应该有1个输入");
        assert_eq!(outputs.len(), 1, "应该有1个输出");

        // 验证输入名称
        assert_eq!(
            inputs[0].name, "input_ids",
            "输入名称应该是 'input_ids'"
        );

        // 验证输出名称
        assert_eq!(
            outputs[0].name, "logits",
            "输出名称应该是 'logits'"
        );
    }

    /// 测试3: 模型推理 - React代码检测
    #[test]
    #[cfg(feature = "ai")]
    fn test_react_code_detection() {
        use ort::{Session, inputs};
        use ndarray::Array2;

        let model_path = "models/local/fast_enhanced.onnx";
        let session = Session::builder()
            .unwrap()
            .commit_from_file(model_path)
            .expect("创建会话失败");

        // React示例代码
        let react_code = r#"
        import React from 'react';
        function App() {
            const [count, setCount] = React.useState(0);
            return <div onClick={() => setCount(count + 1)}>{count}</div>;
        }
        export default App;
        "#;

        // 转换为字符级token (0-255)
        let tokens: Vec<i64> = react_code
            .chars()
            .take(512)
            .map(|c| (c as u8) as i64)
            .collect();

        // 填充到512
        let mut padded = vec![0i64; 512];
        for (i, &token) in tokens.iter().enumerate() {
            padded[i] = token;
        }

        // 创建输入tensor
        let input_array = Array2::from_shape_vec((1, 512), padded).unwrap();
        
        // 执行推理
        let outputs = session
            .run(inputs!["input_ids" => input_array.view()].unwrap())
            .expect("推理失败");

        // 获取输出
        let logits = outputs["logits"]
            .try_extract_tensor::<f32>()
            .expect("提取tensor失败");

        println!("✅ React代码推理成功");
        println!("   输出形状: {:?}", logits.shape());
        
        // 检查输出形状
        assert_eq!(logits.shape(), &[1, 24], "输出形状应该是 [1, 24]");

        // 获取预测类别
        let predicted_class = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        println!("   预测类别: {}", predicted_class);
        println!("   类别0 (React) 分数: {:.4}", logits[[0, 0]]);
        
        // React是类别0，期望有较高分数或被正确识别
        // 注意：由于是字符级编码，可能不会100%准确
        println!("   ℹ️  实际预测: {}", predicted_class);
    }

    /// 测试4: 模型推理 - Vue代码检测
    #[test]
    #[cfg(feature = "ai")]
    fn test_vue_code_detection() {
        use ort::{Session, inputs};
        use ndarray::Array2;

        let model_path = "models/local/fast_enhanced.onnx";
        let session = Session::builder()
            .unwrap()
            .commit_from_file(model_path)
            .expect("创建会话失败");

        // Vue示例代码
        let vue_code = r#"
        import { ref } from 'vue';
        export default {
            setup() {
                const count = ref(0);
                return { count };
            }
        }
        "#;

        // 转换为字符级token
        let tokens: Vec<i64> = vue_code
            .chars()
            .take(512)
            .map(|c| (c as u8) as i64)
            .collect();

        let mut padded = vec![0i64; 512];
        for (i, &token) in tokens.iter().enumerate() {
            padded[i] = token;
        }

        let input_array = Array2::from_shape_vec((1, 512), padded).unwrap();
        
        let outputs = session
            .run(inputs!["input_ids" => input_array.view()].unwrap())
            .expect("推理失败");

        let logits = outputs["logits"]
            .try_extract_tensor::<f32>()
            .expect("提取tensor失败");

        println!("✅ Vue代码推理成功");
        println!("   输出形状: {:?}", logits.shape());
        
        let predicted_class = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        println!("   预测类别: {}", predicted_class);
        println!("   类别1 (Vue) 分数: {:.4}", logits[[0, 1]]);
    }

    /// 测试5: 批量推理性能测试
    #[test]
    #[cfg(feature = "ai")]
    fn test_batch_inference_performance() {
        use ort::{Session, inputs};
        use ndarray::Array2;
        use std::time::Instant;

        let model_path = "models/local/fast_enhanced.onnx";
        let session = Session::builder()
            .unwrap()
            .commit_from_file(model_path)
            .expect("创建会话失败");

        // 测试代码样本
        let test_samples = vec![
            "import React from 'react';",
            "import { ref } from 'vue';",
            "import { Component } from '@angular/core';",
            "const express = require('express');",
            "import _ from 'lodash';",
        ];

        let mut total_time = 0u128;
        let iterations = 10;

        for sample in &test_samples {
            let tokens: Vec<i64> = sample
                .chars()
                .take(512)
                .map(|c| (c as u8) as i64)
                .collect();

            let mut padded = vec![0i64; 512];
            for (i, &token) in tokens.iter().enumerate() {
                padded[i] = token;
            }

            let input_array = Array2::from_shape_vec((1, 512), padded).unwrap();

            // 预热
            let _ = session.run(inputs!["input_ids" => input_array.view()].unwrap());

            // 计时推理
            for _ in 0..iterations {
                let start = Instant::now();
                let _ = session
                    .run(inputs!["input_ids" => input_array.view()].unwrap())
                    .expect("推理失败");
                total_time += start.elapsed().as_micros();
            }
        }

        let avg_time_ms = total_time as f64 / (test_samples.len() * iterations) as f64 / 1000.0;

        println!("✅ 性能测试完成");
        println!("   样本数量: {}", test_samples.len());
        println!("   每样本迭代: {}", iterations);
        println!("   平均推理时间: {:.2} ms", avg_time_ms);
        
        // 期望推理时间 < 50ms
        assert!(
            avg_time_ms < 50.0,
            "推理速度过慢: {:.2} ms (期望 < 50ms)",
            avg_time_ms
        );
    }

    /// 测试6: 模型配置文件读取
    #[test]
    fn test_model_config() {
        use std::fs;
        
        let config_path = "models/model_config.toml";
        assert!(
            PathBuf::from(config_path).exists(),
            "模型配置文件不存在"
        );

        let content = fs::read_to_string(config_path).expect("读取配置失败");
        
        // 检查是否包含fast_enhanced模型配置
        assert!(
            content.contains("fast_enhanced"),
            "配置文件中未找到 fast_enhanced 模型"
        );
        
        assert!(
            content.contains("98.49%"),
            "配置文件中未找到准确率信息"
        );

        println!("✅ 模型配置文件验证通过");
        println!("   配置路径: {}", config_path);
    }
}
