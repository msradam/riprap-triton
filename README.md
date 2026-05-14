# riprap-triton

NVIDIA Triton Inference Server deployment for [Riprap NYC](https://github.com/msradam/riprap-nyc) specialist models.

## Architecture

```
                     ┌─────────────────────────────────────┐
                     │         RunPod L4 GPU Pod           │
                     │                                     │
  HF Space  ──:7860──►   proxy.py  (bearer-auth router)   │
                     │      │                  │           │
                     │      ▼                  ▼           │
                     │  vLLM :8000     tritonserver :8001  │
                     │  Granite 4.1    ┌──────────────┐   │
                     │  8B FP8        │ granite_embed │   │
                     │  (OpenAI API)  │ gliner        │   │
                     │                │ ttm_forecast  │   │
                     │                │ prithvi_pluvial│  │
                     │                │ terramind     │   │
                     │                └──────────────┘   │
                     └─────────────────────────────────────┘
```

**vLLM** handles LLM throughput (Granite 4.1 8B FP8, OpenAI-compatible endpoint).  
**Triton** manages the heterogeneous specialist model lifecycle — warm loading, instance groups, per-model GPU memory, and readiness/liveness probes per model.

## Models

| Triton model | Source | Task |
|---|---|---|
| `granite_embed` | `ibm-granite/granite-embedding-278m-multilingual` | 768-d text embeddings for RAG |
| `gliner` | `urchade/gliner_medium-v2.1` | Typed entity extraction (NER) |
| `ttm_forecast` | `ibm-granite/granite-timeseries-ttm-r2` + fine-tunes | Time-series forecasting (flood recurrence, 311 patterns) |
| `prithvi_pluvial` | `msradam/Prithvi-EO-2.0-NYC-Pluvial` | Sentinel-2 flood segmentation (EO foundation model) |
| `terramind` | `msradam/TerraMind-NYC-Adapters` | LULC / buildings / synthesis via LoRA adapters on TerraMind v1 |

## Proxy API

Same surface as `riprap-nyc/inference-vllm/proxy.py` — all existing HF Space callers work unchanged:

| Endpoint | Routes to |
|---|---|
| `POST /v1/chat/completions` | vLLM :8000 |
| `POST /v1/granite-embed` | Triton `granite_embed` |
| `POST /v1/gliner-extract` | Triton `gliner` |
| `POST /v1/ttm-forecast` | Triton `ttm_forecast` |
| `POST /v1/prithvi-pluvial` | Triton `prithvi_pluvial` |
| `POST /v1/terramind` | Triton `terramind` |
| `GET /healthz` | Fan-out: proxy + vLLM + Triton server + per-model readiness |
| `GET /v1/power` | NVML instantaneous + windowed GPU power (W) |

## Deploy on RunPod

Create a RunPod template:
- **Image**: `nvcr.io/nvidia/tritonserver:25.05-py3`
- **GPU**: L4 (24 GB) or A100
- **Disk**: 50 GB (model weights + pip cache)
- **Ports**: `7860/http, 8001/http, 8003/http` (proxy, Triton HTTP, Triton metrics)
- **Start command**:
  ```bash
  bash -c "curl -fsSL https://raw.githubusercontent.com/msradam/riprap-triton/main/scripts/runpod_triton_setup.sh -o /workspace/setup.sh && bash /workspace/setup.sh > /workspace/setup.log 2>&1; sleep infinity"
  ```
- **Env**: `RIPRAP_PROXY_TOKEN=<token>`, `HF_TOKEN=<hf_token>` (needed for gated model downloads)

## Triton model protocol

All specialist models use Triton's HTTP KFServing v2 protocol. The proxy handles translation — callers use the same riprap REST API they always have.

JSON-in/JSON-out pattern (all models except `granite_embed`):
```
POST /v2/models/{model_name}/infer
{"inputs": [{"name": "request", "shape": [1], "datatype": "BYTES", "data": ["<json_body>"]}]}
```

`granite_embed` takes a flat `BYTES[-1]` array of text strings directly and returns `FP32[-1, 768]` vectors + a JSON `meta` string.

## Healthz

```bash
curl https://<pod>-7860.proxy.runpod.net/healthz
# {
#   "proxy": "ok",
#   "nvml": "ok",
#   "vllm": "ok",
#   "triton": "ok",
#   "triton/granite_embed": "ok",
#   "triton/gliner": "ok",
#   "triton/ttm_forecast": "ok",
#   "triton/prithvi_pluvial": "ok",
#   "triton/terramind": "ok"
# }
```
