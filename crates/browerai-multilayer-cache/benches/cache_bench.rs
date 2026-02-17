use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use browerai_multilayer_cache::{
    CacheLayer, MemoryLayer, MultiLayerCache, PromotePolicy, WriteStrategy,
};

fn bench_single_layer_hit(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let cache: MultiLayerCache<String> = MultiLayerCache::memory_only(Duration::from_secs(60));

    // 预填充数据
    rt.block_on(async {
        for i in 0..100 {
            cache
                .set(&format!("key_{}", i), format!("value_{}", i))
                .await
                .unwrap();
        }
    });

    c.bench_function("L1_hit", |b| {
        b.to_async(&rt).iter(|| async {
            let result = cache.get(black_box("key_50")).await.unwrap();
            black_box(result);
        });
    });
}

fn bench_two_layer_l1_hit(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let l1: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l1"));
    let l2: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l2"));

    let cache = MultiLayerCache::new(vec![l1, l2], Duration::from_secs(60)).unwrap();

    // 预填充 L1
    rt.block_on(async {
        for i in 0..100 {
            cache
                .set(&format!("key_{}", i), format!("value_{}", i))
                .await
                .unwrap();
        }
    });

    c.bench_function("L1_hit_two_layers", |b| {
        b.to_async(&rt).iter(|| async {
            let result = cache.get(black_box("key_50")).await.unwrap();
            black_box(result);
        });
    });
}

fn bench_two_layer_l2_hit(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let l1: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l1"));
    let l2: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l2"));

    let cache = MultiLayerCache::with_options(
        vec![l1.clone(), l2.clone()],
        Duration::from_secs(60),
        WriteStrategy::WriteThrough,
        PromotePolicy::NoPromote, // 不回填，强制每次穿透到 L2
    )
    .unwrap();

    // 只填充 L2
    rt.block_on(async {
        for i in 0..100 {
            l2.set(
                &format!("key_{}", i),
                format!("value_{}", i),
                Duration::from_secs(60),
            )
            .await
            .unwrap();
        }
    });

    c.bench_function("L2_hit_no_promote", |b| {
        b.to_async(&rt).iter(|| async {
            let result = cache.get(black_box("key_50")).await.unwrap();
            black_box(result);
        });
    });
}

fn bench_miss_all_layers(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let l1: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l1"));
    let l2: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l2"));

    let cache = MultiLayerCache::new(vec![l1, l2], Duration::from_secs(60)).unwrap();

    c.bench_function("miss_all_layers", |b| {
        b.to_async(&rt).iter(|| async {
            let result = cache.get(black_box("nonexistent")).await.unwrap();
            black_box(result);
        });
    });
}

fn bench_batch_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let cache: MultiLayerCache<String> = MultiLayerCache::memory_only(Duration::from_secs(60));

    let mut group = c.benchmark_group("batch_operations");

    for &size in [10, 50, 100, 500].iter() {
        group.bench_with_input(BenchmarkId::new("set_batch", size), &size, |b, &size| {
            b.to_async(&rt).iter(|| async {
                let mut batch = HashMap::new();
                for i in 0..size {
                    batch.insert(format!("batch_key_{}", i), format!("batch_value_{}", i));
                }
                cache.set_batch(black_box(batch)).await.unwrap();
            });
        });

        // 预填充数据用于 get_batch
        rt.block_on(async {
            let mut batch = HashMap::new();
            for i in 0..size {
                batch.insert(format!("get_key_{}", i), format!("get_value_{}", i));
            }
            cache.set_batch(batch).await.unwrap();
        });

        group.bench_with_input(BenchmarkId::new("get_batch", size), &size, |b, &size| {
            b.to_async(&rt).iter(|| async {
                let keys: Vec<String> = (0..size).map(|i| format!("get_key_{}", i)).collect();
                let result = cache.get_batch(black_box(&keys)).await.unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_write_strategies(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("write_strategies");

    // WriteThrough
    let l1_through: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l1"));
    let l2_through: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l2"));
    let cache_through = MultiLayerCache::with_options(
        vec![l1_through, l2_through],
        Duration::from_secs(60),
        WriteStrategy::WriteThrough,
        PromotePolicy::PromoteOnHit,
    )
    .unwrap();

    group.bench_function("write_through", |b| {
        b.to_async(&rt).iter(|| async {
            cache_through
                .set(black_box("key"), black_box("value".to_string()))
                .await
                .unwrap();
        });
    });

    // WriteAround
    let l1_around: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l1"));
    let l2_around: Arc<dyn CacheLayer<String>> = Arc::new(MemoryLayer::new("l2"));
    let cache_around = MultiLayerCache::with_options(
        vec![l1_around, l2_around],
        Duration::from_secs(60),
        WriteStrategy::WriteAround,
        PromotePolicy::PromoteOnHit,
    )
    .unwrap();

    group.bench_function("write_around", |b| {
        b.to_async(&rt).iter(|| async {
            cache_around
                .set(black_box("key"), black_box("value".to_string()))
                .await
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_layer_hit,
    bench_two_layer_l1_hit,
    bench_two_layer_l2_hit,
    bench_miss_all_layers,
    bench_batch_operations,
    bench_write_strategies
);

criterion_main!(benches);
