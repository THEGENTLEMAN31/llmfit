# Contributing to llmfit

Thank you for contributing! 🎉

## Quick Start

```bash
# Clone & build
git clone https://github.com/THEGENTLEMAN31/llmfit
cd llmfit
cargo build --release

# Run tests
cargo test --workspace
cargo test -p llmfit-core
cargo test -p llmfit
```

## Development Workflow

1. **Fork & branch**: `git checkout -b feat/amazing-feature`
2. **Make changes** with tests
3. **Run checks**: `cargo fmt --check && cargo clippy --workspace && cargo test --workspace`
4. **Commit**: Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`)
5. **PR**: Open PR with description + linked issue

## Code Style

- **Rust**: `cargo fmt` + `cargo clippy` (deny warnings)
- **JS/React**: ESLint + Prettier (run `npm run lint` in `llmfit-web`)
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)

## Testing

```bash
# All tests
cargo test --workspace

# Core only
cargo test -p llmfit-core

# CLI/TUI
cargo test -p llmfit

# Web tests
cd llmfit-web && npm test
```

## Areas Welcome for Contributions

- **Hardware detection**: New GPU/CPU support, new backends (ROCm, SYCL, Metal)
- **Model catalog**: New model providers, GGUF metadata parsing
- **Scoring**: Better benchmarks, new use-cases, quantization support
- **Web UI**: Dashboard improvements, i18n, accessibility
- **Docs**: Tutorials, examples, API docs
- **Benchmarks**: New hardware profiles, calibration runs

## Code Review Guidelines

- **Be kind**: Constructive, specific, actionable
- **Small PRs**: Easier to review, faster to merge
- **Tests required**: New features need tests
- **Docs updated**: Public APIs need doc comments

## Release Process

Maintainers only: Tag `vX.Y.Z` → GitHub Actions builds/releases automatically.

## Questions?

Open a Discussion or Discord (link in repo description).

---

Thank you for contributing! 🦀
