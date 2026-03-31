#!/usr/bin/env bash
set -euo pipefail

PIPER_VERSION="2023.11.14-2"
MODEL_DIR="${PIPER_MODEL_DIR:-/data/piper-models}"
MODEL_NAME="${PIPER_MODEL_NAME:-en_US-ryan-high}"
PORT="${CODE_CLAW_PORT:-6000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPER_DIR="${SCRIPT_DIR}/bin/piper"
PIPER_BIN="${PIPER_DIR}/piper"

echo "[code-claw] Starting up..."

# 1. Create required directories
mkdir -p /workspace "$MODEL_DIR" "${SCRIPT_DIR}/bin"

# 2. Install piper binary if not present
if [ ! -f "$PIPER_BIN" ]; then
    echo "[code-claw] Installing Piper TTS..."
    PIPER_TMP="${SCRIPT_DIR}/bin/piper_download.tar.gz"
    wget -q "https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_x86_64.tar.gz" \
         -O "$PIPER_TMP"
    tar -xzf "$PIPER_TMP" -C "${SCRIPT_DIR}/bin/"
    chmod +x "$PIPER_BIN"
    rm -f "$PIPER_TMP"
    echo "[code-claw] Piper installed at ${PIPER_BIN}."
fi

# Ensure piper is on PATH
export PATH="${PIPER_DIR}:$PATH"

# 3. Download TTS model if not present
if [ ! -f "${MODEL_DIR}/${MODEL_NAME}.onnx" ]; then
    echo "[code-claw] Downloading Piper voice model ${MODEL_NAME}..."
    BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high"
    wget -q "${BASE_URL}/${MODEL_NAME}.onnx" -O "${MODEL_DIR}/${MODEL_NAME}.onnx"
    wget -q "${BASE_URL}/${MODEL_NAME}.onnx.json" -O "${MODEL_DIR}/${MODEL_NAME}.onnx.json"
    echo "[code-claw] Model downloaded."
fi

# 4. Create virtualenv if needed and install dependencies
VENV="${SCRIPT_DIR}/.venv"
if [ ! -f "${VENV}/bin/python" ]; then
    echo "[code-claw] Creating virtualenv..."
    python3 -m venv "$VENV"
fi
echo "[code-claw] Installing Python dependencies..."
"${VENV}/bin/pip" install --no-cache-dir -q -r "${SCRIPT_DIR}/requirements.txt"

# 5. Launch
echo "[code-claw] Launching on port ${PORT}..."
exec "${VENV}/bin/uvicorn" main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    --app-dir "$SCRIPT_DIR"
