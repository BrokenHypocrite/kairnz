# CUDA + cuDNN base so the ONNX Runtime CUDA provider loads without the torch-lib hack.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential pkg-config git libssl-dev && rm -rf /var/lib/apt/lists/*

# Rust toolchain.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# uv for the Python side.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /kairnz
COPY . .

# Build the Rust self-play/gate binaries and sync the Python env.
RUN cargo build --release -p kairnz-selfplay
RUN cd train && uv sync

CMD ["bash"]
