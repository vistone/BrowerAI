//! Advanced V8 Sandbox with resource limits

use anyhow::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct V8SandboxLimits {
    pub max_heap_mb: usize,
    pub max_execution_time: Duration,
    pub max_call_depth: usize,
}

impl Default for V8SandboxLimits {
    fn default() -> Self {
        Self {
            max_heap_mb: 50,
            max_execution_time: Duration::from_secs(5),
            max_call_depth: 100,
        }
    }
}

pub struct V8Sandbox {
    isolate: v8::OwnedIsolate,
    limits: V8SandboxLimits,
    globals: HashMap<String, String>,
}

impl V8Sandbox {
    pub fn new(limits: V8SandboxLimits) -> Result<Self> {
        let mut params = v8::CreateParams::default();
        params = params.heap_limits(0, limits.max_heap_mb * 1024 * 1024);
        let isolate = v8::Isolate::new(params);
        Ok(Self { isolate, limits, globals: HashMap::new() })
    }

    pub fn set_global(&mut self, key: String, value: String) {
        self.globals.insert(key, value);
    }

    pub fn execute(&mut self, code: &str) -> Result<String> {
        let start_time = Instant::now();
        let handle_scope = &mut v8::HandleScope::new(&mut self.isolate);
        let context = v8::Context::new(handle_scope, Default::default());
        let scope = &mut v8::ContextScope::new(handle_scope, context);

        let global = context.global(scope);
        for (key, value) in &self.globals {
            let key_v8 = v8::String::new(scope, key).unwrap();
            let value_v8 = v8::String::new(scope, value).unwrap();
            global.set(scope, key_v8.into(), value_v8.into());
        }

        let wrapped_code = format!("\"use strict\";\n{}", code);
        let source = v8::String::new(scope, &wrapped_code)
            .ok_or_else(|| anyhow::anyhow!("Failed to create V8 string"))?;

        if start_time.elapsed() > self.limits.max_execution_time {
            return Err(anyhow::anyhow!("Execution time limit exceeded"));
        }

        let script = v8::Script::compile(scope, source, None)
            .ok_or_else(|| anyhow::anyhow!("Failed to compile"))?;
        let result = script.run(scope)
            .ok_or_else(|| anyhow::anyhow!("Failed to execute"))?;
        let result_str = result.to_string(scope)
            .ok_or_else(|| anyhow::anyhow!("Failed to convert result"))?;
        
        Ok(result_str.to_rust_string_lossy(scope))
    }

    pub fn heap_statistics(&mut self) -> V8HeapStats {
        let mut stats = v8::HeapStatistics::default();
        self.isolate.get_heap_statistics(&mut stats);
        V8HeapStats {
            used_heap_size: stats.used_heap_size(),
            total_heap_size: stats.total_heap_size(),
            heap_size_limit: stats.heap_size_limit(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct V8HeapStats {
    pub used_heap_size: usize,
    pub total_heap_size: usize,
    pub heap_size_limit: usize,
}
