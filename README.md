# llmfit

<p align="center">
  <img src="https://img.shields.io/badge/rust-1.85+-orange?logo=rust" alt="Rust version">
  <img src="https://img.shields.io/badge/version-1.1.10-blue" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/github/actions/workflow/status/THEGENTLEMAN31/llmfit/release.yml?label=build" alt="Build">
  <img src="https://img.shields.io/github/actions/workflow/status/THEGENTLEMAN31/llmfit/docker.yml?label=docker" alt="Docker">
  <img src="https://img.shields.io/crates/v/llmfit?label=crates.io" alt="crates.io">
  <img src="https://img.shields.io/github/v/release/THEGENTLEMAN31/llmfit?label=release" alt="release">
</p>

<p align="center">
  <b>Right-size LLM models to your system's hardware</b><br>
  Auto-detects CPU/RAM/GPU → scores 11k+ models → recommends what fits & runs fast
</p>

---

## ✨ Features

- 🔍 **Auto-detection**: CPU, RAM, GPU (NVIDIA/AMD/Apple Silicon), PCIe, NVLink
- 📊 **Multi-objective scoring**: Quality + Speed + VRAM fit + Context + Energy + Cost
- ⚡ **Live calibration**: Community benchmarks → your hardware
- 📋 **Hardware plans**: llama.cpp / vLLM commands with `-ngl`, `--n-cpu-moe`, TP
- 🌐 **Dashboard**: Web UI + CLI + REST API + TUI
- 📦 **Model catalog**: 11k+ models from HF + GGUF introspection

## 🚀 Quick Start

```bash
# Install (choose one)
cargo install llmfit                    # crates.io
brew install THEGENTLEMAN31/tap/llmfit  # Homebrew
docker run ghcr.io/thegentleman31/llmfit  # Docker
# or download binary from GitHub Releases

# Detect hardware & find models
llmfit system          # Show hardware specs
llmfit fit --perfect -n 10   # Top 10 perfect-fit models
llmfit fit --use-case coding -n 10  # Coding models

# Plan hardware for a specific model
llmfit plan "Qwen/Qwen2.5-7B-Instruct" --context 8192

# Web dashboard
llmfit serve  # or: cd llmfit-web && npx vite dev
```

## 📊 Example Output

```
$ llmfit fit --perfect -n 3

=== System Specifications ===
CPU: Intel i5-10200H (8 cores)
RAM: 38.9 GB (avail 19.2 GB)
GPU: NVIDIA RTX 3050 Laptop (4.0 GB VRAM)

╭─────────┬────────────────────────────┬────────┬──────┬───────┬─────────┬───────────┬────────┬──────┬───────┐
│ Status  │ Model                      │ Provider │ Size │ Score │ tok/s   │ Quant     │ Runtime │ Mode │ Mem % │
├─────────┼────────────────────────────┼──────────┼──────┼───────┼─────────┼───────────┼─────────┼──────┼───────┤
│ 🟢      │ Qwen2.5-Coder-3B-AWQ       │ Alibaba  │ 3.4B │ 85    │ 85.5    │ AWQ-4bit  │ vLLM    │ GPU  │ 62%   │
│ 🟢      │ Qwen2.5-3B-AWQ             │ Alibaba  │ 3.4B │ 85    │ 85.5    │ AWQ-4bit  │ vLLM    │ GPU  │ 62%   │
│ 🟢      │ Qwen2.5-3B-GPTQ-Int4       │ Alibaba  │ 3.1B │ 85    │ 94.1    │ GPTQ-Int4 │ vLLM    │ GPU  │ 58%   │
╰─────────┴────────────────────────────┴──────────┴──────┴───────┴─────────┴───────────┴─────────┴──────┴───────┘
```

## 🎯 Use Cases

| Use Case | Command |
|----------|---------|
| **Coding assistant** | `llmfit fit --use-case coding --perfect -n 5` |
| **Chat/Chatbot** | `llmfit fit --use-case chat -n 10` |
| **Reasoning/Logic** | `llmfit fit --use-case reasoning --perfect` |
| **Local LLM server** | `llmfit serve --host 0.0.0.0 --port 8787` |
| **Benchmark** | `llmfit bench --model "Qwen2.5-7B" --runtime llamacpp` |

## 🐳 Docker

```bash
docker run --gpus all -p 8787:8787 ghcr.io/thegentleman31/llmfit serve --host 0.0.0.0
# Web dashboard: http://localhost:8787
```

## 🛠 Installation

| Method | Command |
|--------|---------|
| **Cargo** | `cargo install llmfit` |
| **Homebrew** | `brew install THEGENTLEMAN31/tap/llmfit` |
| **Docker** | `docker run ghcr.io/thegentleman31/llmfit` |
| **Binary** | [GitHub Releases](https://github.com/THEGENTLEMAN31/llmfit/releases) |

### Cargo Features
```bash
cargo install llmfit --features "web,bench"  # Optional features
```

## 🏗 Building from Source

```bash
git clone https://github.com/THEGENTLEMAN31/llmfit
cd llmfit
cargo build --release
./target/release/llmfit --help
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) - PRs welcome!

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - inference engine
- [gguf](https://github.com/ggerganov/ggml) - model format
- [localmaxxing.com](https://localmaxxing.com) - community benchmarks
- All model authors on HuggingFace
