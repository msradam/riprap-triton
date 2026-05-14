"""Riprap Triton Proxy — bearer-auth gateway on port 7860.

Routes:
  /v1/chat/completions  →  vLLM      :8000   (OpenAI-compat)
  /v1/completions       →  vLLM      :8000
  /v1/models            →  vLLM      :8000
  /v1/granite-embed     →  Triton    :8001  model=granite_embed
  /v1/gliner-extract    →  Triton    :8001  model=gliner
  /v1/ttm-forecast      →  Triton    :8001  model=ttm_forecast
  /v1/prithvi-pluvial   →  Triton    :8001  model=prithvi_pluvial
  /v1/terramind         →  Triton    :8001  model=terramind
  /healthz              →  fan-out check of all upstreams
  /v1/power             →  GPU power via NVML (L4)

Triton translation:
  All specialist endpoints use Triton HTTP KFServing v2 inference
  (/v2/models/{name}/infer). Structured inputs use TYPE_BYTES tensors
  carrying the raw JSON body, outputs are JSON strings parsed back.
  This keeps the proxy thin — no per-model schema logic beyond routing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

log = logging.getLogger("riprap.triton_proxy")

VLLM_URL   = "http://127.0.0.1:8000"
TRITON_URL = "http://127.0.0.1:8001"

PROXY_TOKEN = os.environ.get("RIPRAP_PROXY_TOKEN", "")

# Triton model name for each riprap endpoint
_TRITON_ROUTE: dict[str, str] = {
    "/v1/granite-embed":   "granite_embed",
    "/v1/gliner-extract":  "gliner",
    "/v1/ttm-forecast":    "ttm_forecast",
    "/v1/prithvi-pluvial": "prithvi_pluvial",
    "/v1/terramind":       "terramind",
}

app = FastAPI(title="Riprap Triton Proxy")

# ---------------------------------------------------------------------------
# GPU power sampler (NVML)
# ---------------------------------------------------------------------------

_SAMPLES: deque[tuple[float, float]] = deque(maxlen=600)
_SAMPLER_TASK: asyncio.Task | None = None
_NVML_OK = False
_NVML_HANDLE = None
_NVML_ERR: str | None = None


def _init_nvml() -> None:
    global _NVML_OK, _NVML_HANDLE, _NVML_ERR
    try:
        import pynvml
        pynvml.nvmlInit()
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE)
        _NVML_OK = True
        log.info("NVML initialized")
    except Exception as e:
        _NVML_ERR = f"{type(e).__name__}: {e}"
        _NVML_OK = False
        log.warning("NVML init failed: %s", _NVML_ERR)


def _read_power_w() -> float | None:
    if not _NVML_OK:
        return None
    try:
        import pynvml
        return pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0
    except Exception:
        return None


async def _power_sampler() -> None:
    while True:
        p = _read_power_w()
        if p is not None:
            _SAMPLES.append((time.time(), p))
        await asyncio.sleep(0.1)


def _avg_power_over(t0: float, t1: float) -> float | None:
    if not _SAMPLES:
        return None
    bucket = [p for ts, p in _SAMPLES if t0 <= ts <= t1]
    return sum(bucket) / len(bucket) if bucket else (_SAMPLES[-1][1] if _SAMPLES else None)


@app.on_event("startup")
async def _startup() -> None:
    _init_nvml()
    if _NVML_OK:
        global _SAMPLER_TASK
        _SAMPLER_TASK = asyncio.create_task(_power_sampler())


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _SAMPLER_TASK:
        _SAMPLER_TASK.cancel()
    if _NVML_OK:
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _check_auth(request: Request) -> None:
    if not PROXY_TOKEN:
        raise HTTPException(503, "RIPRAP_PROXY_TOKEN not configured")
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    if auth.removeprefix("Bearer ").strip() != PROXY_TOKEN:
        raise HTTPException(401, "invalid bearer token")


# ---------------------------------------------------------------------------
# Triton KFServing v2 translation
# ---------------------------------------------------------------------------


def _make_triton_request(body_bytes: bytes, input_name: str = "request") -> dict:
    """Wrap raw JSON body into a Triton v2 infer request with a BYTES tensor."""
    return {
        "inputs": [
            {
                "name": input_name,
                "shape": [1],
                "datatype": "BYTES",
                "data": [body_bytes.decode("utf-8")],
            }
        ],
        "outputs": [{"name": "result"}],
    }


def _make_embed_triton_request(body_bytes: bytes) -> dict:
    """Granite-embed takes a BYTES array of individual texts, not JSON."""
    payload = json.loads(body_bytes)
    texts = payload.get("texts", [])
    return {
        "inputs": [
            {
                "name": "texts",
                "shape": [len(texts)],
                "datatype": "BYTES",
                "data": texts,
            }
        ],
        "outputs": [{"name": "vectors"}, {"name": "meta"}],
    }


def _make_gliner_triton_request(body_bytes: bytes) -> dict:
    payload = json.loads(body_bytes)
    return {
        "inputs": [
            {
                "name": "text",
                "shape": [1],
                "datatype": "BYTES",
                "data": [payload.get("text", "")],
            },
            {
                "name": "labels",
                "shape": [len(payload.get("labels", []))],
                "datatype": "BYTES",
                "data": payload.get("labels", []),
            },
        ],
        "outputs": [{"name": "result"}],
    }


async def _call_triton(model_name: str, triton_req: dict,
                       timeout: float = 300.0) -> dict:
    url = f"{TRITON_URL}/v2/models/{model_name}/infer"
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=triton_req)
        r.raise_for_status()
        return r.json()


def _extract_triton_result(triton_resp: dict) -> dict:
    """Pull the JSON result string out of a Triton v2 inference response."""
    for output in triton_resp.get("outputs", []):
        if output["name"] == "result":
            raw = output["data"][0]
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, bytes):
                return json.loads(raw.decode())
    return triton_resp


def _extract_embed_result(triton_resp: dict, meta_str: str | None) -> dict:
    """Reconstruct granite-embed response from Triton outputs."""
    vectors = None
    meta = {}
    for output in triton_resp.get("outputs", []):
        if output["name"] == "vectors":
            vectors = output["data"]
        elif output["name"] == "meta":
            raw = output["data"][0]
            try:
                meta = json.loads(raw if isinstance(raw, str) else raw.decode())
            except Exception:
                pass
    result = dict(meta)
    result["vectors"] = vectors
    if vectors is not None:
        result.setdefault("ok", True)
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {"service": "riprap-triton", "ok": True, "nvml": _NVML_OK,
            "nvml_err": None if _NVML_OK else _NVML_ERR}


@app.get("/healthz")
async def healthz():
    out = {
        "proxy": "ok",
        "nvml": "ok" if _NVML_OK else f"err: {_NVML_ERR}",
    }
    async with httpx.AsyncClient(timeout=5) as client:
        # vLLM
        try:
            r = await client.get(f"{VLLM_URL}/health")
            out["vllm"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
        except Exception as e:
            out["vllm"] = f"err: {type(e).__name__}"
        # Triton readiness
        try:
            r = await client.get(f"{TRITON_URL}/v2/health/ready")
            out["triton"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
        except Exception as e:
            out["triton"] = f"err: {type(e).__name__}"
        # Per-model liveness
        for endpoint, model in _TRITON_ROUTE.items():
            try:
                r = await client.get(f"{TRITON_URL}/v2/models/{model}/ready")
                out[f"triton/{model}"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
            except Exception as e:
                out[f"triton/{model}"] = f"err: {type(e).__name__}"
    return out


@app.get("/v1/power")
async def power(request: Request) -> Response:
    _check_auth(request)
    if not _NVML_OK:
        return JSONResponse({"ok": False, "err": _NVML_ERR or "NVML unavailable"}, status_code=503)
    now = time.time()
    inst = _read_power_w()
    return JSONResponse({
        "ok": True, "ts": now,
        "power_w": inst,
        "power_w_avg_1s": _avg_power_over(now - 1.0, now),
        "power_w_avg_5s": _avg_power_over(now - 5.0, now),
        "samples_held": len(_SAMPLES),
        "device": "NVIDIA L4",
    })


# vLLM passthrough

async def _stream_passthrough(upstream: httpx.Response) -> AsyncIterator[bytes]:
    async for chunk in upstream.aiter_raw():
        yield chunk


async def _proxy_vllm(path: str, request: Request, timeout: float = 300.0) -> Response:
    body = await request.body()
    headers = {
        "content-type": request.headers.get("content-type", "application/json"),
        "accept": request.headers.get("accept", "*/*"),
    }
    is_stream = b'"stream":true' in body or b'"stream": true' in body
    client = httpx.AsyncClient(timeout=timeout)
    t0 = time.time()
    upstream_req = client.build_request("POST", f"{VLLM_URL}{path}", content=body, headers=headers)
    upstream = await client.send(upstream_req, stream=is_stream)
    if is_stream:
        snap = _read_power_w()
        return StreamingResponse(
            _stream_passthrough(upstream),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers={"x-gpu-power-w": f"{snap:.2f}" if snap else "", "x-gpu-stream": "1"},
            background=upstream.aclose,
        )
    content = await upstream.aread()
    await upstream.aclose()
    await client.aclose()
    t1 = time.time()
    extra = {}
    if _NVML_OK:
        avg_w = _avg_power_over(t0, t1)
        if avg_w:
            extra = {
                "x-gpu-power-w": f"{avg_w:.3f}",
                "x-gpu-energy-j": f"{avg_w * (t1 - t0):.3f}",
                "x-gpu-duration-s": f"{t1 - t0:.3f}",
                "x-gpu-device": "NVIDIA L4",
            }
    return Response(content=content, status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"),
                    headers=extra)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    _check_auth(request)
    return await _proxy_vllm("/v1/chat/completions", request)


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    _check_auth(request)
    return await _proxy_vllm("/v1/completions", request)


@app.get("/v1/models")
async def models(request: Request) -> Response:
    _check_auth(request)
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{VLLM_URL}/v1/models")
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))


# Triton specialist routes

@app.post("/v1/granite-embed")
async def granite_embed(request: Request) -> Response:
    _check_auth(request)
    body = await request.body()
    try:
        triton_req = _make_embed_triton_request(body)
        triton_resp = await _call_triton("granite_embed", triton_req)
        result = _extract_embed_result(triton_resp, None)
    except Exception as e:
        result = {"ok": False, "err": f"{type(e).__name__}: {e}"}
    return JSONResponse(result)


@app.post("/v1/gliner-extract")
async def gliner_extract(request: Request) -> Response:
    _check_auth(request)
    body = await request.body()
    try:
        triton_req = _make_gliner_triton_request(body)
        triton_resp = await _call_triton("gliner", triton_req)
        result = _extract_triton_result(triton_resp)
    except Exception as e:
        result = {"ok": False, "err": f"{type(e).__name__}: {e}"}
    return JSONResponse(result)


@app.post("/v1/ttm-forecast")
async def ttm_forecast(request: Request) -> Response:
    _check_auth(request)
    body = await request.body()
    try:
        triton_req = _make_triton_request(body, input_name="request")
        triton_resp = await _call_triton("ttm_forecast", triton_req, timeout=120.0)
        result = _extract_triton_result(triton_resp)
    except Exception as e:
        result = {"ok": False, "err": f"{type(e).__name__}: {e}"}
    return JSONResponse(result)


@app.post("/v1/prithvi-pluvial")
async def prithvi_pluvial(request: Request) -> Response:
    _check_auth(request)
    body = await request.body()
    try:
        triton_req = _make_triton_request(body, input_name="request")
        triton_resp = await _call_triton("prithvi_pluvial", triton_req, timeout=120.0)
        result = _extract_triton_result(triton_resp)
    except Exception as e:
        result = {"ok": False, "err": f"{type(e).__name__}: {e}"}
    return JSONResponse(result)


@app.post("/v1/terramind")
async def terramind(request: Request) -> Response:
    _check_auth(request)
    body = await request.body()
    try:
        triton_req = _make_triton_request(body, input_name="request")
        triton_resp = await _call_triton("terramind", triton_req, timeout=180.0)
        result = _extract_triton_result(triton_resp)
    except Exception as e:
        result = {"ok": False, "err": f"{type(e).__name__}: {e}"}
    return JSONResponse(result)


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    """OpenAI-compat embeddings alias → granite_embed."""
    _check_auth(request)
    body = await request.body()
    try:
        triton_req = _make_embed_triton_request(body)
        triton_resp = await _call_triton("granite_embed", triton_req)
        result = _extract_embed_result(triton_resp, None)
    except Exception as e:
        result = {"ok": False, "err": f"{type(e).__name__}: {e}"}
    return JSONResponse(result)
