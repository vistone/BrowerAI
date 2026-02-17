use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use browerai_deobfuscation::{
    SymbolicExecutor, DataFlowAnalyzer, TypeInferencer, AdvancedDeobfuscationPipeline,
    ControlFlowAnalyzer, StringPoolExtractor, ObfuscationPatternLibrary,
};

// Sample code snippets of varying complexity
fn get_test_samples() -> Vec<(&'static str, &'static str)> {
    vec![
        ("small", r#"var x = 42; function foo() { return x + 10; }"#),
        ("medium", r#"
            var _0xarr = ['log', 'hello', 'world'];
            console[_0xarr[0]](_0xarr[1] + ' ' + _0xarr[2]);
            function calc(a, b) { return a + b; }
            var x = 0x10 + 0x20;
        "#),
        ("large", r#"
            var _0x5a = String['fromCharCode'](72,101,108,108,111);
            var _0xb = ['\x48\x65\x6c\x6c\x6f'];
            var _0xc = 'He' + 'llo';
            function calc(a, b) {
                if (!(a > b)) { return a + b; }
                while (a > 0) { a--; if (a == 0) break; }
                return b;
            }
            console['log'](_0x5a);
            eval('console.log("test")');
            var arr = [1,2,3,4,5].map(x => x * 2);
        "#),
    ]
}

fn bench_symbolic_executor(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbolic_executor");
    
    for (name, code) in get_test_samples() {
        group.bench_with_input(BenchmarkId::from_parameter(name), code, |b, code| {
            b.iter(|| {
                let mut executor = SymbolicExecutor::new();
                executor.analyze(black_box(code))
            });
        });
    }
    
    group.finish();
}

fn bench_data_flow_analyzer(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_flow_analyzer");
    
    for (name, code) in get_test_samples() {
        group.bench_with_input(BenchmarkId::from_parameter(name), code, |b, code| {
            b.iter(|| {
                let mut analyzer = DataFlowAnalyzer::new();
                analyzer.analyze(black_box(code))
            });
        });
    }
    
    group.finish();
}

fn bench_type_inferencer(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_inferencer");
    
    for (name, code) in get_test_samples() {
        group.bench_with_input(BenchmarkId::from_parameter(name), code, |b, code| {
            b.iter(|| {
                let mut inferencer = TypeInferencer::new();
                inferencer.infer(black_box(code))
            });
        });
    }
    
    group.finish();
}

fn bench_control_flow_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("control_flow_graph");
    
    for (name, code) in get_test_samples() {
        group.bench_with_input(BenchmarkId::from_parameter(name), code, |b, code| {
            b.iter(|| {
                let mut cfg = ControlFlowAnalyzer::new();
                let _ = cfg.build_cfg(black_box(code));
                cfg.reachability_analysis();
                cfg.detect_loops()
            });
        });
    }
    
    group.finish();
}

fn bench_string_pool_extractor(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_pool_extractor");
    
    for (name, code) in get_test_samples() {
        group.bench_with_input(BenchmarkId::from_parameter(name), code, |b, code| {
            b.iter(|| {
                let mut extractor = StringPoolExtractor::new();
                let _ = extractor.extract(black_box(code));
                extractor.get_statistics()
            });
        });
    }
    
    group.finish();
}

fn bench_obfuscation_pattern_library(c: &mut Criterion) {
    let mut group = c.benchmark_group("obfuscation_pattern_library");
    
    for (name, code) in get_test_samples() {
        group.bench_with_input(BenchmarkId::from_parameter(name), code, |b, code| {
            b.iter(|| {
                let library = ObfuscationPatternLibrary::new();
                library.detect(black_box(code))
            });
        });
    }
    
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    
    for (name, code) in get_test_samples() {
        group.bench_with_input(BenchmarkId::from_parameter(name), code, |b, code| {
            b.iter(|| {
                let pipeline = AdvancedDeobfuscationPipeline::new();
                pipeline.process(black_box(code))
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_symbolic_executor,
    bench_data_flow_analyzer,
    bench_type_inferencer,
    bench_control_flow_graph,
    bench_string_pool_extractor,
    bench_obfuscation_pattern_library,
    bench_full_pipeline,
);

criterion_main!(benches);
