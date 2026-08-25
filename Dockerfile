# Multi-stage build - use Rust 1.95 to support current dependencies
FROM rust:1.95-slim-bookworm AS builder

WORKDIR /app
COPY . .

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libudev-dev \
    && rm -rf /var/lib/apt/lists/*

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/llmfit /usr/local/bin/llmfit

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash llmfit
USER llmfit

WORKDIR /home/llmfit

ENTRYPOINT ["llmfit"]
CMD ["--help"]
