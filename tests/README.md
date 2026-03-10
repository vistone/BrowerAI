# BrowerAI Integration Tests

This directory contains integration tests that verify the interaction between multiple crates in the BrowerAI project.

## Test Categories

### 1. Parser Integration Tests
- `test_html_parsing_integration` - HTML parsing and DOM building
- `test_css_parsing_integration` - CSS parsing and stylesheet creation
- `test_js_parsing_integration` - JavaScript parsing and AST generation
- `test_parser_pipeline` - Chain multiple parsers together

### 2. Error Handling Tests
- `test_error_handling_integration` - Graceful handling of malformed input

### 3. Performance Tests
- `test_large_document_parsing` - Performance with large documents
- `test_thread_safety` - Concurrent parsing safety
- `test_memory_efficiency` - Memory management verification

### 4. End-to-End Tests
- `test_full_page_processing` - Complete page parsing workflow
- `test_end_to_end_workflow` - Full pipeline: parse → render → analyze

### 5. Feature Integration Tests
- `test_feature_extraction_integration` - AI feature extraction (requires `ai` feature)
- `test_ai_core_integration` - AI core functionality (requires `ai` feature)
- `test_renderer_integration` - Rendering pipeline
- `test_devtools_integration` - Developer tools functionality

## Running Tests

```bash
# Run all integration tests
cargo test --test integration_test

# Run with AI features
cargo test --test integration_test --features ai

# Run specific test
cargo test --test integration_test test_end_to_end_workflow

# Run with output
cargo test --test integration_test -- --nocapture
```

## Test Data

Tests use inline test data for simplicity. For more complex scenarios, consider:
- Adding fixtures in `tests/fixtures/`
- Using `tempfile` for temporary files
- Mocking external dependencies

## CI Integration

These tests are designed to run in CI environments:
- No external network dependencies
- Deterministic results
- Reasonable execution time (< 30 seconds total)
- Thread-safe execution
