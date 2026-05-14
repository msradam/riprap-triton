#!/usr/bin/env bash
# RunPod startup for riprap-triton.
# Base image: nvcr.io/nvidia/tritonserver:25.05-py3
#
# Architecture:
#   tritonserver  :8001  (all specialist ML models as Python backends)
#   vLLM          :8000  (Granite 4.1 8B FP8, localhost-only, OpenAI-compat)
#   proxy.py      :7860  (bearer-auth router: vLLM + Triton)
#
# The script is idempotent — safe to re-run in the same container.

REPO_URL="https://github.com/msradam/riprap-triton.git"
REPO_DIR="/workspace/riprap-triton"
LOG_DIR="/workspace/logs"
HF_CACHE="/workspace/hf_cache"
PROXY_TOKEN="${RIPRAP_PROXY_TOKEN:-pm7AmssTxoi0OSvXZH6ciMDwOATzlwXjPHBUKJ-cjQk}"

mkdir -p "$LOG_DIR" "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

trap 'echo "[setup] ERROR at line $LINENO — container stays alive"; sleep infinity' ERR

# --- [1] Clone / update repo -------------------------------------------------
echo "==> [1/6] Clone riprap-triton"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only
else
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

# --- [2] Python deps for Triton backends + proxy -----------------------------
echo "==> [2/6] Install Python deps"
# Triton image ships Python 3.10 + torch; use its pip directly.
pip install --quiet \
    "fastapi>=0.115" "uvicorn[standard]>=0.32" "httpx>=0.27" "pydantic>=2.9" \
    "nvidia-ml-py>=12.560" \
    "sentence-transformers>=5.0.0" \
    "gliner>=0.2.6" \
    "granite-tsfm==0.3.3" \
    "peft==0.18.1" \
    "huggingface_hub>=0.34" \
    "safetensors>=0.4" \
    "einops" "tifffile" "albumentations" "scipy"

# --- [3] vLLM ----------------------------------------------------------------
echo "==> [3/6] Install vLLM"
pip install --quiet "vllm==0.7.3" "numpy<2.0"
pip cache purge 2>/dev/null || true

# --- [4] terratorch (EO stack, best-effort) ----------------------------------
echo "==> [4/6] Install terratorch (EO stack)"
# terratorch has conflicts with system packages (cryptography, blinker).
# Install with --ignore-installed to skip uninstall of distutils packages.
pip install --quiet "blinker>=1.8" "cryptography>=41" --ignore-installed 2>/dev/null || true
pip install --quiet \
    "terratorch==1.1rc6" "diffusers" "timm" \
    "segmentation-models-pytorch" "kornia" \
    --ignore-installed 2>/dev/null || \
    echo "    WARN: terratorch install failed — Prithvi/TerraMind will skip"
pip cache purge 2>/dev/null || true
df -h / | tail -1 | awk '{print "    disk after installs: "$3" used, "$4" free"}'

# --- [5] Start tritonserver --------------------------------------------------
echo "==> [5/6] Start tritonserver on :8001"
pkill -f tritonserver 2>/dev/null || true

# tritonserver ships at /opt/tritonserver/bin/tritonserver
TRITON_BIN="/opt/tritonserver/bin/tritonserver"
MODEL_REPO="$REPO_DIR/model_repository"

nohup "$TRITON_BIN" \
    --model-repository="$MODEL_REPO" \
    --http-port=8001 \
    --grpc-port=8002 \
    --metrics-port=8003 \
    --log-verbose=0 \
    --log-info=1 \
    --model-control-mode=none \
    --exit-on-error=false \
    > "$LOG_DIR/triton.log" 2>&1 &

echo "    tritonserver pid $! — waiting up to 300s for readiness..."
TRITON_OK=0
for i in $(seq 1 300); do
    if curl -sf http://127.0.0.1:8001/v2/health/ready > /dev/null 2>&1; then
        echo "    tritonserver ready after ${i}s"; TRITON_OK=1; break
    fi
    sleep 1
done
[ "$TRITON_OK" -eq 0 ] && echo "    WARN: tritonserver not ready — proxy will start anyway"

# --- [6a] Start vLLM ---------------------------------------------------------
echo "==> [6a/6] Start vLLM on :8000 (localhost-only)"
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
mkdir -p /tmp/prometheus_multiproc
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc

nohup python -m vllm.entrypoints.openai.api_server \
    --model ibm-granite/granite-4.1-8b-fp8 \
    --served-model-name granite4.1:8b \
    --host 127.0.0.1 \
    --port 8000 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 4096 \
    --enforce-eager \
    --disable-log-requests \
    > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
echo "    vLLM pid $VLLM_PID — waiting up to 240s for health..."
VLLM_OK=0
for i in $(seq 1 240); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "    vLLM ready after ${i}s"; VLLM_OK=1; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "    ERROR: vLLM died — see $LOG_DIR/vllm.log"; break
    fi
    sleep 1
done
[ "$VLLM_OK" -eq 0 ] && echo "    WARN: vLLM not healthy — proxy will still start"

# --- [6b] Start proxy (foreground) -------------------------------------------
echo "==> [6b/6] Start proxy on :7860 (foreground)"
pkill -f "proxy:app" 2>/dev/null || true
cp "$REPO_DIR/proxy/proxy.py" /workspace/proxy.py
export RIPRAP_PROXY_TOKEN="$PROXY_TOKEN"
cd /workspace

echo
echo "Stack: tritonserver :8001  vLLM :8000  proxy :7860"
echo "Logs:  $LOG_DIR/"
echo

exec uvicorn proxy:app --host 0.0.0.0 --port 7860 --log-level info
