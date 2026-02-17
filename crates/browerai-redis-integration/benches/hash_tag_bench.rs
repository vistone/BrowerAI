//! Redis 集群哈希标签基准测试
//!
//! 评估哈希标签在批量操作中的性能和正确性。

use browerai_redis_integration::ClusterHashTag;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_hash_tag_apply(c: &mut Criterion) {
    c.bench_function("hash_tag_apply_single", |b| {
        b.iter(|| {
            ClusterHashTag::apply(black_box("user:1:profile:name"), black_box(Some("user:1")))
        })
    });

    c.bench_function("hash_tag_apply_batch_100", |b| {
        let keys: Vec<String> = (0..100).map(|i| format!("key:{}", i)).collect();
        b.iter(|| ClusterHashTag::apply_batch(black_box(&keys), black_box(Some("bucket:1"))))
    });
}

fn benchmark_hash_tag_strip(c: &mut Criterion) {
    c.bench_function("hash_tag_strip_with_tag", |b| {
        b.iter(|| ClusterHashTag::strip(black_box("user:1:profile{user:1}")))
    });

    c.bench_function("hash_tag_strip_without_tag", |b| {
        b.iter(|| ClusterHashTag::strip(black_box("simple_key")))
    });
}

fn benchmark_hash_tag_extract(c: &mut Criterion) {
    c.bench_function("hash_tag_extract_present", |b| {
        b.iter(|| ClusterHashTag::extract(black_box("user:1:profile{user:1}")))
    });

    c.bench_function("hash_tag_extract_absent", |b| {
        b.iter(|| ClusterHashTag::extract(black_box("simple_key")))
    });
}

fn benchmark_batch_workflow(c: &mut Criterion) {
    c.bench_function("batch_apply_strip_extract", |b| {
        let keys: Vec<String> = (0..50).map(|i| format!("data:{}:value", i)).collect();
        b.iter(|| {
            let tagged =
                ClusterHashTag::apply_batch(black_box(&keys), black_box(Some("session:abc")));
            for t in &tagged {
                let _ = ClusterHashTag::strip(t);
                let _ = ClusterHashTag::extract(t);
            }
        })
    });
}

criterion_group!(
    benches,
    benchmark_hash_tag_apply,
    benchmark_hash_tag_strip,
    benchmark_hash_tag_extract,
    benchmark_batch_workflow
);
criterion_main!(benches);
