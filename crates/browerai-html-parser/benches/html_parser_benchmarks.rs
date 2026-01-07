use browerai_html_parser::HtmlParser;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn benchmark_simple_html(c: &mut Criterion) {
    let parser = HtmlParser::new();
    let html = "<html><body><h1>Hello World</h1></body></html>";

    c.bench_function("parse simple html", |b| {
        b.iter(|| parser.parse(black_box(html)))
    });
}

fn benchmark_complex_html(c: &mut Criterion) {
    let parser = HtmlParser::new();
    let html = r#"
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Test Page</title>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>Welcome</h1>
                    <nav>
                        <ul>
                            <li><a href="/">Home</a></li>
                            <li><a href="/about">About</a></li>
                        </ul>
                    </nav>
                </header>
                <main>
                    <article>
                        <h2>Article Title</h2>
                        <p>This is a paragraph with <strong>bold</strong> and <em>italic</em> text.</p>
                    </article>
                </main>
                <footer>
                    <p>&copy; 2026 BrowerAI</p>
                </footer>
            </div>
        </body>
        </html>
    "#;

    c.bench_function("parse complex html", |b| {
        b.iter(|| parser.parse(black_box(html)))
    });
}

fn benchmark_nested_divs(c: &mut Criterion) {
    let parser = HtmlParser::new();

    let mut group = c.benchmark_group("nested_divs");
    for depth in [5, 10, 20].iter() {
        let mut html = String::new();
        for _ in 0..*depth {
            html.push_str("<div>");
        }
        html.push_str("Content");
        for _ in 0..*depth {
            html.push_str("</div>");
        }

        group.bench_with_input(BenchmarkId::from_parameter(depth), depth, |b, _| {
            b.iter(|| parser.parse(black_box(&html)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_simple_html,
    benchmark_complex_html,
    benchmark_nested_divs
);
criterion_main!(benches);
