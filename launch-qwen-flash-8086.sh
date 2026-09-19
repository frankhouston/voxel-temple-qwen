#!/usr/bin/env bash
# launch-qwen-flash-8086.sh
# Launch Qwen3.8-Flash-Next V3 on OpenAI-compatible endpoint (port 8086)
#
# Model: Qwen3.8-Flash-Next V3
# Params: 177B MoE (512 experts, Q4_0)
# Hardware: M1 Max 64GB (Metal GPU AGXMetalG13X)
# Fork: npanj/llama.cpp (MoE SSD streaming)
#
# Usage:
#   ./launch-qwen-flash-8086.sh
#
# After startup, the API is available at:
#   http://localhost:8086/v1/chat/completions
#
# Example request:
#   curl http://localhost:8086/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Hello"}],"stream":true}'

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.8-Flash-Next}"
BINARY="${BINARY:-$HOME/llama.cpp-npanj/build/bin/llama-server}"
PORT="${PORT:-8086}"
HOST="${HOST:-0.0.0.0}"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: llama-server binary not found at $BINARY"
    echo "Build it first: cd $HOME/llama.cpp-npanj && LLAMA_METAL=1 LLAMA_BUILD_SERVER=ON cmake -B build -DGGML_METAL=ON && cmake --build build --config Release"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download Qwen3.8-Flash-Next V3 (Q4_0, 95.5 GiB) to $MODEL_PATH"
    exit 1
fi

echo "=== Qwen3.8-Flash-Next V3 → OpenAI Endpoint ==="
echo "Model:  $MODEL_PATH"
echo "Binary: $BINARY"
echo "Host:   $HOST:$PORT"
echo "GPU:    Metal (AGXMetalG13X) via --gpu-layers 99"
echo "MoE:    SSD streaming enabled (--moe-stream)"
echo "I/O:    8 parallel threads for NVMe throughput"
echo "CTX:    32K tokens (--ctx-size 32768)"
echo "MTP:    Draft head enabled (--spec-type draft-mtp)"
echo "Cache:  Prompt caching enabled (--cache-prompt)"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec "$BINARY" \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --moe-stream \
    --moe-stream-capacity 8 \
    --io-threads 8 \
    --gpu-layers 99 \
    --ctx-size 32768 \
    --batch-size 2048 \
    --threads "$(sysctl -n hw.ncpu)" \
    --flash-attn \
    --cache-prompt \
    --spec-type draft-mtp \
    --temp 0.0 \
    --top-p 0.95 \
    --no-warn
