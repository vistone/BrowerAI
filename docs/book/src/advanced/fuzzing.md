# Fuzz Testing

Fuzz testing is an automated testing technique that provides random data as input to find bugs, crashes, and security vulnerabilities.

## Overview

BrowerAI uses cargo-fuzz with libFuzzer for continuous fuzz testing of all parsers.

## Setup

### Install cargo-fuzz

```bash
cargo install cargo-fuzz
```

### Fuzz Targets

BrowerAI includes three fuzz targets:
- `fuzz_html_parser` - HTML parsing
- `fuzz_css_parser` - CSS parsing  
- `fuzz_js_parser` - JavaScript parsing

## Running Fuzz Tests

### HTML Parser

```bash
cd fuzz
cargo fuzz run fuzz_html_parser
```

### CSS Parser

```bash
cd fuzz
cargo fuzz run fuzz_css_parser
```

### JavaScript Parser

```bash
cd fuzz
cargo fuzz run fuzz_js_parser
```

## Options

### Time-Limited Runs

```bash
cargo fuzz run fuzz_html_parser -- -max_total_time=60
```

### Multiple Workers

```bash
cargo fuzz run fuzz_html_parser -- -workers=4
```

### With Corpus

```bash
cargo fuzz run fuzz_html_parser corpus/html_parser
```

## Crash Investigation

### Finding Crashes

Crashes are saved in:
```
fuzz/artifacts/fuzz_html_parser/
fuzz/artifacts/fuzz_css_parser/
fuzz/artifacts/fuzz_js_parser/
```

### Reproducing

```bash
cargo fuzz run fuzz_html_parser fuzz/artifacts/fuzz_html_parser/crash-xxxxx
```

### Minimizing

```bash
cargo fuzz tmin fuzz_html_parser fuzz/artifacts/fuzz_html_parser/crash-xxxxx
```

## CI Integration

Fuzz tests run in CI with time limits:

```yaml
- name: Run Fuzz Tests
  run: |
    cd fuzz
    cargo fuzz run fuzz_html_parser -- -max_total_time=300
```

## Best Practices

1. **Start with short runs** to verify setup
2. **Use multiple workers** for speed
3. **Monitor memory** to avoid OOM
4. **Save interesting inputs** to corpus
5. **Integrate with CI** for continuous fuzzing

## Resources

- [cargo-fuzz book](https://rust-fuzz.github.io/book/cargo-fuzz.html)
- [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html)
- [Fuzz testing guide](https://github.com/rust-fuzz/trophy-case)
