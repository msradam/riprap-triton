"""Triton Python backend — Granite TTM r2 time-series forecasting.

Request/response both travel as JSON strings (BYTES[1]) to handle
the variable-length history array and multi-model dispatch cleanly.

Input request JSON:
  model: str          # zero_shot_battery | fine_tune_battery | weekly_311 | floodnet_recurrence
  history: [float]
  context_length: int
  prediction_length: int
  cadence: str        # default "h"

Output result JSON:
  ok, elapsed_s, device, model, forecast:[float], peak_index, peak_value
"""
import json
import time
from threading import Lock

import numpy as np
import triton_python_backend_utils as pb_utils

_TTM_MODELS = {
    "zero_shot_battery":    "ibm-granite/granite-timeseries-ttm-r2",
    "fine_tune_battery":    "msradam/Granite-TTM-r2-Battery-Surge",
    "weekly_311":           "ibm-granite/granite-timeseries-ttm-r2",
    "floodnet_recurrence":  "ibm-granite/granite-timeseries-ttm-r2",
}


class TritonPythonModel:
    _instances: dict = {}
    _lock = Lock()
    _device = "cpu"

    def initialize(self, args):
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Warm all models at startup so first request is instant.
        for key in _TTM_MODELS:
            self._get_model(key)
        pb_utils.Logger.log_info("ttm_forecast: all variants ready")

    def _get_model(self, model_key: str):
        if model_key in self._instances:
            return self._instances[model_key]
        with self._lock:
            if model_key in self._instances:
                return self._instances[model_key]
            import torch
            pb_utils.Logger.log_info(f"ttm_forecast: loading {model_key}")
            if model_key == "fine_tune_battery":
                from huggingface_hub import snapshot_download
                from tsfm_public import TinyTimeMixerForPrediction
                local_dir = snapshot_download(_TTM_MODELS[model_key])
                m = TinyTimeMixerForPrediction.from_pretrained(local_dir).eval()
            else:
                from tsfm_public.toolkit.get_model import get_model
                m = get_model(
                    _TTM_MODELS[model_key],
                    context_length=512,
                    prediction_length=96,
                ).eval()
            if self._device == "cuda" and torch.cuda.is_available():
                m = m.to("cuda")
            self._instances[model_key] = m
            return m

    def execute(self, requests):
        responses = []
        for request in requests:
            t0 = time.time()
            raw = pb_utils.get_input_tensor_by_name(request, "request").as_numpy()
            payload = json.loads(
                raw.flatten()[0].decode("utf-8")
                if isinstance(raw.flatten()[0], bytes)
                else raw.flatten()[0]
            )

            model_key = payload.get("model", "zero_shot_battery")
            history = payload["history"]
            ctx_len = int(payload["context_length"])
            pred_len = int(payload["prediction_length"])
            cadence = payload.get("cadence", "h")

            try:
                import torch
                m = self._get_model(model_key)
                series = np.array(history, dtype="float32")
                if len(series) < ctx_len:
                    pad = np.full(
                        ctx_len - len(series),
                        series[0] if len(series) else 0.0,
                        dtype="float32",
                    )
                    series = np.concatenate([pad, series])
                series = series[-ctx_len:]
                x = torch.from_numpy(series).float().unsqueeze(0).unsqueeze(-1)
                if self._device == "cuda":
                    x = x.to("cuda")
                with torch.no_grad():
                    out = m(past_values=x)
                fc = out.prediction_outputs.squeeze(-1).squeeze(0).cpu().numpy()
                peak_idx = int(np.argmax(np.abs(fc)))
                result = {
                    "ok": True,
                    "elapsed_s": round(time.time() - t0, 3),
                    "device": self._device,
                    "model": model_key,
                    "context_length": ctx_len,
                    "prediction_length": pred_len,
                    "cadence": cadence,
                    "forecast": [round(float(v), 6) for v in fc.tolist()],
                    "peak_index": peak_idx,
                    "peak_value": round(float(fc[peak_idx]), 6),
                }
            except Exception as e:
                result = {"ok": False, "err": f"{type(e).__name__}: {e}", "model": model_key}

            out_t = pb_utils.Tensor(
                "result",
                np.array([json.dumps(result).encode()], dtype=object),
            )
            responses.append(pb_utils.InferenceResponse([out_t]))
        return responses

    def finalize(self):
        self._instances.clear()
