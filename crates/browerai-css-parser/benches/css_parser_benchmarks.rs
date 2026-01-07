use browerai_css_parser::CssParser;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_simple_css(c: &mut Criterion) {
    let parser = CssParser::new();
    let css = "body { color: red; margin: 0; }";

    c.bench_function("parse simple css", |b| {
        b.iter(|| parser.parse(black_box(css)))
    });
}

fn benchmark_complex_css(c: &mut Criterion) {
    let parser = CssParser::new();
    let css = r#"
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1, h2, h3 {
            color: #333;
            margin-bottom: 10px;
        }
        
        a:hover {
            color: #007bff;
            text-decoration: underline;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
        }
    "#;

    c.bench_function("parse complex css", |b| {
        b.iter(|| parser.parse(black_box(css)))
    });
}

fn benchmark_validate_css(c: &mut Criterion) {
    let parser = CssParser::new();
    let css = "div { margin: 10px; padding: 5px; }";

    c.bench_function("validate css", |b| {
        b.iter(|| parser.validate(black_box(css)))
    });
}

criterion_group!(
    benches,
    benchmark_simple_css,
    benchmark_complex_css,
    benchmark_validate_css
);
criterion_main!(benches);
