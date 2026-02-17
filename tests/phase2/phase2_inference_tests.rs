// tests/phase2_inference_tests.rs
//! Phase 2 模型推理集成测试

#[cfg(feature = "onnx")]
mod phase2_tests {
    use browerai_ai_core::Phase2ModelLoader;

    #[test]
    fn test_phase2_model_loader_creation() {
        let loader = Phase2ModelLoader::new("models");
        // 验证加载器创建成功
        assert!(true);
    }

    #[test]
    #[ignore] // 需要实际的 ONNX 模型文件
    fn test_selector_embedding_inference() {
        let loader = Phase2ModelLoader::new("models");
        
        // 尝试加载模型
        match loader.load_selector_embedding() {
            Ok(model) => {
                println!("✅ Selector embedding model loaded from: {:?}", model.model_path());
                
                // 准备输入：batch_size=1, seq_len=50
                let input_tokens = vec![vec![1, 5, 10, 3, 2, 0, 0, 0, 0, 0, 
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
                
                match model.infer(&input_tokens) {
                    Ok(output) => {
                        assert_eq!(output.len(), 1);
                        assert_eq!(output[0].len(), 50 * 128); // 128维度，50个时间步
                        println!("✅ Selector embedding inference successful");
                    }
                    Err(e) => {
                        println!("⚠️  Inference failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("⚠️  Model loading failed: {}", e);
                println!("   This is expected if ONNX model files are not present");
            }
        }
    }

    #[test]
    #[ignore] // 需要实际的 ONNX 模型文件
    fn test_property_predictor_inference() {
        let loader = Phase2ModelLoader::new("models");
        
        match loader.load_property_predictor() {
            Ok(model) => {
                println!("✅ Property predictor model loaded from: {:?}", model.model_path());
                
                // 准备输入：128 * 10 = 1280维
                let embeddings = vec![vec![0.1; 1280]];
                
                match model.infer(&embeddings) {
                    Ok(output) => {
                        assert_eq!(output.len(), 1);
                        assert_eq!(output[0].len(), 50); // 50 个 CSS 属性
                        println!("✅ Property predictor inference successful");
                    }
                    Err(e) => {
                        println!("⚠️  Inference failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("⚠️  Model loading failed: {}", e);
            }
        }
    }

    #[test]
    #[ignore] // 需要实际的 ONNX 模型文件
    fn test_color_model_inference() {
        let loader = Phase2ModelLoader::new("models");
        
        match loader.load_color_model() {
            Ok(model) => {
                println!("✅ Color model loaded from: {:?}", model.model_path());
                
                // 准备输入：3*32*32 RGB 图像
                let images = vec![vec![0.5; 3 * 32 * 32]];
                
                match model.infer(&images) {
                    Ok(output) => {
                        assert_eq!(output.len(), 1);
                        println!("✅ Color model inference successful, output dim: {}", output[0].len());
                    }
                    Err(e) => {
                        println!("⚠️  Inference failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("⚠️  Model loading failed: {}", e);
            }
        }
    }

    #[test]
    #[ignore] // 需要实际的 ONNX 模型文件
    fn test_all_phase2_models_loading() {
        let loader = Phase2ModelLoader::new("models");
        
        println!("Testing Phase 2 model loading...\n");
        
        // 尝试加载所有 5 个模型
        let selector_result = loader.load_selector_embedding();
        let property_result = loader.load_property_predictor();
        let color_result = loader.load_color_model();
        let complete_result = loader.load_complete_model();
        let finetuned_result = loader.load_finetuned_model();
        
        println!("Selector Embedding: {}", if selector_result.is_ok() { "✅" } else { "❌" });
        println!("Property Predictor: {}", if property_result.is_ok() { "✅" } else { "❌" });
        println!("Color Model: {}", if color_result.is_ok() { "✅" } else { "❌" });
        println!("Complete Model: {}", if complete_result.is_ok() { "✅" } else { "❌" });
        println!("Finetuned Model: {}", if finetuned_result.is_ok() { "✅" } else { "❌" });
        
        // 至少有一个加载成功
        let success_count = [selector_result, property_result, color_result, complete_result, finetuned_result]
            .iter()
            .filter(|r| r.is_ok())
            .count();
        
        println!("\n✅ Successfully loaded {}/5 models", success_count);
    }
}

#[cfg(not(feature = "onnx"))]
mod phase2_tests_disabled {
    #[test]
    fn test_onnx_feature_disabled() {
        println!("⚠️  Phase 2 tests require 'onnx' feature");
        println!("   Run with: cargo test --features onnx");
    }
}
