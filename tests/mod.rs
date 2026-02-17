// Consolidated integration test modules

#[path = "ai/ai_fallback_tests.rs"]
mod ai_fallback_tests;
#[path = "ai/ai_integration_tests.rs"]
mod ai_integration_tests;

#[path = "deobfuscation/deobfuscation_controlflow_tests.rs"]
mod deobfuscation_controlflow_tests;
#[path = "deobfuscation/deobfuscation_encoding_tests.rs"]
mod deobfuscation_encoding_tests;
#[path = "deobfuscation/deobfuscation_transform_tests.rs"]
mod deobfuscation_transform_tests;
#[path = "deobfuscation/deobfuscation_variables_functions_tests.rs"]
mod deobfuscation_variables_functions_tests;

#[path = "e2e/e2e_integration_tests.rs"]
mod e2e_integration_tests;
#[path = "e2e/e2e_website_tests.rs"]
mod e2e_website_tests;

#[path = "framework/framework_detection_tests.rs"]
mod framework_detection_tests;

#[path = "js/js_compatibility_tests.rs"]
mod js_compatibility_tests;
#[path = "js/jsunpack_integration_tests.rs"]
mod jsunpack_integration_tests;
#[path = "js/unified_js_interface_integration_tests.rs"]
mod unified_js_interface_integration_tests;

#[path = "phase2/css_phase2_integration_tests.rs"]
mod css_phase2_integration_tests;
#[path = "phase2/phase2_day4_5_integration_tests.rs"]
mod phase2_day4_5_integration_tests;
#[path = "phase2/phase2_inference_tests.rs"]
mod phase2_inference_tests;
#[path = "phase2/phase2_integration_tests.rs"]
mod phase2_integration_tests;

#[path = "phase3/phase3_day3_4_integration_tests.rs"]
mod phase3_day3_4_integration_tests;
#[path = "phase3/phase3_week2_integration_tests.rs"]
mod phase3_week2_integration_tests;
#[path = "phase3/phase3_week3_enhanced_call_graph_tests.rs"]
mod phase3_week3_enhanced_call_graph_tests;

#[path = "monitoring/fast_enhanced_integration_tests.rs"]
mod fast_enhanced_integration_tests;
#[path = "monitoring/monitoring_integration_tests.rs"]
mod monitoring_integration_tests;

#[path = "integration/comprehensive_integration_tests.rs"]
mod comprehensive_integration_tests;
#[path = "integration/rust_python_framework_integration_tests.rs"]
mod rust_python_framework_integration_tests;
#[path = "integration/step4_rust_integration_tests.rs"]
mod step4_rust_integration_tests;
