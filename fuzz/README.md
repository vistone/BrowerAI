# Fuzz Testing Guide

This directory contains fuzz testing targets for BrowerAI using cargo-fuzz and libFuzzer.

## Prerequisites

Install cargo-fuzz:
```bash
cargo install cargo-fuzz
```

## Running Fuzz Tests

### HTML Parser Fuzzing
```bash
cd fuzz
cargo fuzz run fuzz_html_parser
```

### CSS Parser Fuzzing
```bash
cd fuzz
cargo fuzz run fuzz_css_parser
```

### JavaScript Parser Fuzzing
```bash
cd fuzz
cargo fuzz run fuzz_js_parser
```

## Running with Specific Options

```bash
# Run for 60 seconds
cargo fuzz run fuzz_html_parser -- -max_total_time=60

# Run with specific number of workers
cargo fuzz run fuzz_html_parser -- -workers=4

# Run with corpus directory
cargo fuzz run fuzz_html_parser corpus/html_parser
```

## CI Integration

Fuzz tests can be run in CI with time limits:
```bash
cargo fuzz run fuzz_html_parser -- -max_total_time=300 -runs=1000000
```

## Finding Crashes

When crashes are found, they are saved in:
```
fuzz/artifacts/fuzz_html_parser/
fuzz/artifacts/fuzz_css_parser/
fuzz/artifacts/fuzz_js_parser/
```

## Reproducing Crashes

```bash
cargo fuzz run fuzz_html_parser fuzz/artifacts/fuzz_html_parser/crash-xxxxx
```

## Corpus Management

The corpus contains test cases that have been discovered by the fuzzer:
- `corpus/html_parser/` - HTML test cases
- `corpus/css_parser/` - CSS test cases
- `corpus/js_parser/` - JavaScript test cases

## Minimizing Crash Cases

```bash
cargo fuzz cmin fuzz_html_parser
cargo fuzz tmin fuzz_html_parser fuzz/artifacts/fuzz_html_parser/crash-xxxxx
```

## Best Practices

1. **Start with short runs** to verify the setup works
2. **Use multiple workers** to speed up fuzzing
3. **Monitor memory** usage to avoid OOM
4. **Save interesting inputs** to corpus for regression testing
5. **Integrate with CI** for continuous fuzzing

## Target Details

### fuzz_html_parser
- Tests HTML parsing robustness
- Ensures no panics on malformed HTML
- Tests text extraction

### fuzz_css_parser
- Tests CSS parsing robustness
- Ensures no panics on malformed CSS
- Tests validation logic

### fuzz_js_parser
- Tests JavaScript parsing robustness
- Ensures no panics on malformed JS
- Tests validation logic

## Resources

- [cargo-fuzz book](https://rust-fuzz.github.io/book/cargo-fuzz.html)
- [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html)
- [Fuzz testing best practices](https://github.com/rust-fuzz/trophy-case)
