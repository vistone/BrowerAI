# Unmaintained Dependencies

## Overview

This document tracks unmaintained dependencies in the project that are accepted as transitive dependencies with no immediate security impact. These advisories are intentionally ignored in [deny.toml](../deny.toml) until upstream alternatives become available.

## Status: January 2026

### Current Unmaintained Crates (4)

| Crate | Version | RustSec ID | Dependency Chain | Status |
|-------|---------|------------|------------------|--------|
| `fxhash` | 0.2.1 | RUSTSEC-2025-0057 | selectors → browerai-css-parser | ✅ Tracked |
| `number_prefix` | 0.4.0 | RUSTSEC-2025-0119 | indicatif → criterion (dev) | ✅ Tracked |
| `paste` | 1.0.15 | RUSTSEC-2024-0436 | boa_* → browerai-dom/js-parser | ✅ Tracked |
| `rustls-pemfile` | 1.0.4 | RUSTSEC-2025-0134 | reqwest → browerai-network | ✅ Tracked |

### Validation Results

- **Security Vulnerabilities**: 0 (zero)
- **Unmaintained Warnings**: 4 (expected, documented above)
- **cargo deny check**: ✅ PASS (all subsystems)
- **cargo audit**: ✅ No vulnerabilities

## Dependency Details

### 1. fxhash (RUSTSEC-2025-0057)

**Status**: Unmaintained, archived repository  
**Why we have it**: Transitive via `selectors` (CSS parser dependency)  
**Security impact**: None - hashing algorithm for internal use  
**Alternative**: `rustc-hash`  
**Waiting for**: cssparser/selectors upstream update

```
selectors v0.25.0 → fxhash v0.2.1
└── browerai-css-parser
```

### 2. number_prefix (RUSTSEC-2025-0119)

**Status**: Unmaintained  
**Why we have it**: Transitive via `indicatif` (progress bar, dev dependency via criterion)  
**Security impact**: None - formatting library for benchmarks  
**Alternative**: `unit-prefix`  
**Waiting for**: criterion/indicatif upstream update

```
criterion v0.5.1 → indicatif v0.17.11 → number_prefix v0.4.0
└── browerai-css-parser (dev-dependencies)
```

**Note**: cargo-deny currently doesn't detect RUSTSEC-2025-0119 (database sync issue), but cargo-audit does.

### 3. paste (RUSTSEC-2024-0436)

**Status**: Archived (Oct 6, 2024)  
**Why we have it**: Transitive via `boa` JavaScript engine (v0.20.0)  
**Security impact**: None - procedural macro for token pasting  
**Alternative**: `pastey` (when boa upgrades)  
**Waiting for**: boa v0.21+ upgrade (breaking changes)

```
boa_string v0.20.0 → paste v1.0.15 (proc-macro)
├── boa_ast → boa_engine → browerai-dom
└── browerai-js-parser
```

**Usage**: 256k+ projects still use paste; macro expansion happens at compile time.

### 4. rustls-pemfile (RUSTSEC-2025-0134)

**Status**: Unmaintained, archived (August 2025)  
**Why we have it**: Transitive via `reqwest` (HTTP client)  
**Security impact**: None - PEM parsing for TLS  
**Alternative**: `rustls-pki-types` (PemObject trait)  
**Waiting for**: reqwest upgrade to use rustls-pki-types API

```
reqwest v0.11.27 → rustls-pemfile v1.0.4
├── browerai-network
└── browerai-learning (HTTP requests)
```

## Monitoring & Maintenance

### Automated Checks

1. **Pre-commit validation** (`scripts/pre-commit.sh`):
   - Runs `cargo deny check advisories` (with ignore list)
   - Runs `cargo audit` (reports but doesn't fail on unmaintained)

2. **GitHub Actions** (on pull requests):
   - cargo-audit check reports unmaintained warnings
   - Alerts on new security vulnerabilities (none currently)

### Manual Review Schedule

- **Monthly**: Check for upstream updates (boa, selectors, reqwest, criterion)
- **Quarterly**: Evaluate feasibility of direct dependency upgrades
- **On release**: Verify no new unmaintained dependencies introduced

### How to Update

When upstream alternatives become available:

1. Update dependency version in relevant `Cargo.toml`
2. Remove corresponding entry from `deny.toml` ignore list
3. Run `cargo update` and test
4. Verify `cargo deny check advisories` passes without warnings
5. Update this document

## References

- [RustSec Advisory Database](https://rustsec.org/)
- [deny.toml Configuration](../deny.toml)
- [Pre-commit Checks Documentation](./PRE_COMMIT_CHECKS.md)

## Decision Rationale

These unmaintained crates are accepted because:

1. **No security vulnerabilities**: cargo-audit confirms zero CVEs
2. **Transitive only**: Not direct dependencies we control
3. **Stable functionality**: Mature crates unlikely to need updates
4. **Wide usage**: paste (256k projects), rustls-pemfile (standard TLS stack)
5. **Upstream blockers**: Waiting for boa 0.21, reqwest updates, etc.

---

**Last Updated**: 2026-01-08  
**Next Review**: 2026-02-08 (monthly upstream check)
