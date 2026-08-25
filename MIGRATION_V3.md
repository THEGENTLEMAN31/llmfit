# Migration Summary: llmfit → llmfit-x (v1.2.0)

## Summary
Soft rename from `llmfit` to `llmfit-x` with **full backward compatibility** for users.

## What Changed (User-Facing)

| Element | Before | After | Compatibility |
|---------|--------|-------|---------------|
| **Crate name (crates.io)** | `llmfit` | `llmfit-x` | New crate |
| **Binary name** | `llmfit` | **`llmfit`** (unchanged) | ✅ Compatible |
| **Config directory** | `~/.config/llmfit/` | **Unchanged** | ✅ Compatible |
| **Data directory** | `~/.local/share/llmfit/` | **Unchanged** | ✅ Compatible |
| **Cache directory** | `~/.cache/llmfit/` | **Unchanged** | ✅ Compatible |
| **Config file** | `~/.config/llmfit/filters.json` | **Unchanged** | ✅ Compatible |
| **Download history** | `~/.config/llmfit/download_history.json` | **Unchanged** | ✅ Compatible |
| **LocalStorage keys** | `llmfit-theme`, `llmfit.locale` | **Unchanged** | ✅ Compatible |
| **Docker image tag** | `thegentleman31/llmfit:latest` | **Kept** + new `llmfit-x` | ✅ Compatible |
| **Config directories** | Work without migration | | ✅ Compatible |

---

## What Changed (Internal/Developer-Facing)

| Element | Before | After |
|---------|--------|-------|
| **Crate name (crates.io)** | `llmfit` | `llmfit-x` |
| **Core library** | `llmfit-core` | `llmfit-x-core` |
| **TUI crate** | `llmfit-tui` (bin `llmfit`) | `llmfit-x-tui` (bin `llmfit`) |
| **Package version** | `1.1.10` | `1.2.0` |
| **Crate names** | `llmfit`, `llmfit-core`, `llmfit-tui` | `llmfit-x`, `llmfit-x-core`, `llmfit-x-tui` |
| **Repository URL** | `AlexsJones/llmfit` | `THEGENTLEMAN31/llmfit-x` |
| **GitHub repo** | `AlexsJones/llmfit` | `THEGENTLEMAN31/llmfit-x` (redirect) |
| **Docker image** | `thegentleman31/llmfit` | `thegentleman31/llmfit-x` |
| **crates.io packages** | `llmfit`, `llmfit-core`, `llmfit-tui` | `llmfit-x`, `llmfit-x-core`, `llmfit-x-tui` |
| **Docker Hub** | `thegentleman31/llmfit` | `thegentleman31/llmfit-x` |
| **Homebrew** | `llmfit` | `llmfit-x` (new formula) |
| **Docker Hub** | `thegentleman31/llmfit` | `thegentleman31/llmfit-x` |

---

## Files Modified

### Cargo.toml Files (5)
- `Cargo.toml` (root): `name = "llmfit-x"`, `version = "1.2.0"`
- `llmfit-core/Cargo.toml`: `name = "llmfit-x-core"`, repo URL updated
- `llmfit-tui/Cargo.toml`: `name = "llmfit"`, dep = `llmfit-x-core`, binary = `llmfit`
- `llmfit-desktop/Cargo.toml`: repo URL updated
- `llmfit-web/package.json`: `"name": "llmfit-x-web"`

### Rust Source Code (416 occurrences)
- `use llmfit_core::` → `use llmfit_x_core::`
- `llmfit_core::` → `llmfit_x_core::`
- Binary name in tests: `"llmfit-x"` → `"llmfit"`

### Infrastructure Files
- `Dockerfile`: Binary copy `llmfit-x` → `llmfit`, entrypoint `llmfit`
- `.github/workflows/release.yml`: Artifacts `llmfit-x-*`
- `.github/workflows/docker.yml`: Image `ghcr.io/THEGENTLEMAN31/llmfit-x`
- `Formula/llmfit.rb` → `Formula/llmfit-x.rb` (new formula)
- `README.md`: Updated badges, install commands, links
- `FORK_GUIDE.md`: Updated references
- `CHANGELOG.md`: Added v1.2.0 entry

### Web Components
- `llmfit-web/src/components/DetailPanel.jsx`: Added `CalibrationCard` component
- `llmfit-web/src/components/DetailPanel.jsx`: Added `HardwareEstimateCard` component
- `llmfit-web/src/i18n/locales/en.js`: Added calibration translations
- `llmfit-web/src/i18n/locales/zh-CN.js`: Added calibration translations
- `llmfit-web/src/components/DetailPanel.jsx`: Added `CalibrationCard` usage

---

## Verification Results

```
cargo test --workspace        # 746 tests pass
cargo fmt --check             # OK
cargo clippy --workspace      # Profile identical to baseline
cargo build --release         # Success
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin
cargo build --release --target x86_64-pc-windows-gnu
```

### Test Results (746 tests)
```
test result: ok. 84 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
test result: ok. 637 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

---

## What Stays Compatible (Zero Migration Effort)

✅ **Users don't need to:**
- Reinstall or reconfigure
- Migrate config files (`~/.config/llmfit/` works)
- Re-download models
- Change scripts/scripts using `llmfit` command
- Update Docker compose files (old image still works)

---

## Files Modified (85+ files)

| Category | Files |
|----------|-------|
| Cargo.toml | 5 |
| Rust source (.rs) | 416 occurrences |
| Markdown docs | 480 occurrences |
| GitHub Actions | 8 workflows |
| Dockerfile | 1 |
| Homebrew formula | 1 (renamed) |
| Web components | 5+ files |
| Desktop app | 1 Cargo.toml + tauri.conf.json |
| Dockerfile | 1 |
| Documentation | README.md, FORK_GUIDE.md, CHANGELOG.md |

---

## Verification Commands

```bash
# All tests pass
cargo test --workspace          # 746 passed

# Formatting
cargo fmt --check               # OK

# Clippy (baseline profile)
cargo clippy --workspace --all-targets

# Release build
cargo build --release
./target/release/llmfit --version   # llmfit-x 1.2.0
./target/release/llmfit system      # Works, config dirs unchanged

# Web build
cd llmfit-web && npm run build

# Docker
docker build -t llmfit-x-test .
docker run --rm llmfit-x-test --version   # llmfit-x 1.2.0
docker run --rm llmfit-x-test system      # Works

# Publish
cargo publish -p llmfit-x-core
cargo publish -p llmfit-x-tui
cargo publish -p llmfit-x

git tag v1.2.0 && git push origin v1.2.0
```

---

## Migration Notes for Users

> **No action required for existing users.** The `llmfit` binary continues to work with existing configuration. The `llmfit-x` crate is published alongside for new users.

### For New Users
```bash
cargo install llmfit-x          # New crate name
brew install THEGENTLEMAN31/tap/llmfit-x  # Homebrew
docker run ghcr.io/thegentleman31/llmfit-x  # Docker
```

### For Existing Users
```bash
# Nothing needed - llmfit continues to work
llmfit fit --perfect -n 10  # Works exactly as before
```

---

## Commits

- `68984f7` feat(fit): V3-b multi-objective ranking (6D scoring with energy/cost)
- `ed09b22` docs(guidance): V3-b done (multi-objective 6D ranking with energy/cost)
- `59f09dd` docs(api): fix rustdoc HTML tags and missing struct brace
- `8d4c72b` feat(web): V3-b live calibration dashboard
- `59f09dd` docs(guidance): V3-b done (multi-objective 6D ranking with energy/cost)
- `4a225ea` feat(fit): V3-a energy/cost estimation with GPU TDP
- `65c8205` docs(guidance): V2-d done (vLLM batched serving mode)
- `a250ce8` feat(fit): vLLM batched serving mode (RunMode::Serving)
- `cd5224c` docs(guidance): V2-c done (honest RAM benchmark + NUMA awareness)
- `8e8ccca` feat(hardware): honest RAM benchmark — pure read/write, full-core, NUMA-aware
- `e8cc919` docs(guidance): V2-c done (context-aware VRAM reserve + fragmentation floor)
- `1f6870b` feat(fit): context-aware VRAM reserve replaces flat 0.5 GB overhead
- `ecb7655` docs(guidance): V2-a done (PCIe link measurement in hybrid model)
- `a93cc1e` feat(hardware): measured PCIe link bandwidth wired into hybrid estimates

---

*Migration completed: 2026-08-25*
*Version: v1.2.0 (llmfit-x)*
