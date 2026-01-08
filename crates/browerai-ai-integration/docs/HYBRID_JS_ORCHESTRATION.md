# Hybrid JavaScript Orchestration (V8 + SWC + Boa)

This document describes the hybrid JavaScript orchestration layer that combines:

- V8: for high-performance, highly compatible parsing/compilation and execution
- SWC: for rich AST extraction with TypeScript/JSX/module support
- Boa: a pure-Rust parser/runtime for secure fallback and lightweight validation

## Overview

The orchestrator chooses the best engine per task:

- Parsing/AST: Prefer SWC; fallback to Boa (script mode)
- Execution: Prefer V8; fallback to Boa engine in-process sandbox
- Validation: Uses the same engine selected for parsing

## Policy Control

Environment variable BROWERAI_JS_POLICY controls strategy:

- performance: V8 + SWC whenever available
- secure: Prefer Boa execution; use V8 only when necessary
- balanced (default): Dynamic selection based on code features

Optional .env example:

```dotenv
BROWERAI_JS_POLICY=balanced
```

## Feature Heuristics

Lightweight detection on the source string:

- Modules: presence of `import`/`export`
- TypeScript: `: type`, `interface`, `type` keywords
- JSX: `<Tag>` and closing `</...>` patterns

These influence the parser choice and the reported `SourceKind`.

## API

Types are re-exported from `browerai-ai-integration`:

- `HybridJsOrchestrator`
- `OrchestrationPolicy`
- `UnifiedAst` with `AstEngine` and `SourceKind`

Basic usage:

```rust
use browerai_ai_integration::{HybridJsOrchestrator, OrchestrationPolicy};

let mut orch = HybridJsOrchestrator::with_policy(OrchestrationPolicy::Balanced);
let ast = orch.parse("import x from 'y';").unwrap();
let result = orch.execute("1 + 2").unwrap();
```

## Notes

- V8 initialization failures gracefully fall back to Boa.
- Boa execution is a placeholder (string: ok) pending hardened sandbox integration; validation is performed via parsing.
- SWC extraction currently uses a heuristic extractor; full AST traversal can be enabled incrementally.
