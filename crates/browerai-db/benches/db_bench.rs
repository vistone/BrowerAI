use browerai_db::{DbConnection, DbOperations};
use criterion::{criterion_group, criterion_main, Criterion};
use std::env;

fn bench_set_get(c: &mut Criterion) {
    let Some(url) = env::var("DATABASE_URL").ok() else {
        eprintln!("Skipping benchmarks: DATABASE_URL not set");
        return;
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let conn = match rt.block_on(DbConnection::new(&url, 10)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping benchmarks: cannot connect ({}).", e);
            return;
        }
    };
    if let Err(e) = rt.block_on(conn.init_tables()) {
        eprintln!("Skipping benchmarks: init tables failed ({}).", e);
        return;
    }
    let ops = DbOperations::new(conn);

    c.bench_function("db_set_get", |b| {
        b.iter(|| {
            rt.block_on(async {
                let key = format!("bench_key_{}", rand::random::<u64>());
                let val = vec![7u8; 256];
                ops.set(&key, val.clone(), 120).await.unwrap();
                let got = ops.get(&key).await.unwrap();
                assert_eq!(got, Some(val));
            })
        })
    });
}

fn bench_set_get_batch(c: &mut Criterion) {
    let Some(url) = env::var("DATABASE_URL").ok() else {
        eprintln!("Skipping benchmarks: DATABASE_URL not set");
        return;
    };

    let rt = tokio::runtime::Runtime::new().unwrap();
    let conn = match rt.block_on(DbConnection::new(&url, 10)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping benchmarks: cannot connect ({}).", e);
            return;
        }
    };
    if let Err(e) = rt.block_on(conn.init_tables()) {
        eprintln!("Skipping benchmarks: init tables failed ({}).", e);
        return;
    }
    let ops = DbOperations::new(conn);

    c.bench_function("db_set_get_batch", |b| {
        b.iter(|| {
            rt.block_on(async {
                let entries: Vec<_> = (0..50)
                    .map(|i| {
                        let key = format!("bench_batch_key_{}_{}", i, rand::random::<u32>());
                        let val = vec![i as u8; 32];
                        (key, val, 300)
                    })
                    .collect();

                ops.set_batch(&entries).await.unwrap();

                let keys: Vec<String> = entries.iter().map(|(k, _, _)| k.clone()).collect();
                let got = ops.get_batch(&keys).await.unwrap();
                assert_eq!(got.len(), entries.len());
            })
        })
    });
}

criterion_group!(benches, bench_set_get, bench_set_get_batch);
criterion_main!(benches);
