# Benchmarks and the engine for the N5105 (or any x86-64 Linux box), as a container.
# Build from the repository top folder:
#   docker build -t moe-bench -f docker/Dockerfile .
FROM rust:1-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends fio python3 python3-numpy \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /kit
COPY kernel /kit/kernel
COPY ssd /kit/ssd
COPY engine /kit/engine
COPY trace /kit/trace
COPY docker/run_all.sh /kit/run_all.sh

# kernels detect SSSE3/SSE4.1 at runtime, so a generic x86-64 build is correct
RUN cd /kit/kernel && cargo build --release \
 && cp target/release/int8bench /usr/local/bin/int8bench && rm -rf target \
 && cd /kit/engine && cargo build --release \
 && cp target/release/moe-engine /usr/local/bin/moe-engine && rm -rf target \
 && rm -rf /usr/local/cargo/registry \
 && chmod +x /kit/run_all.sh /kit/ssd/bench_ssd.sh

ENTRYPOINT ["/kit/run_all.sh"]
