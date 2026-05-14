"""Triton Python backend — Prithvi-EO-2.0-NYC-Pluvial flood segmentation.

Request/response as JSON strings (BYTES[1]) — S2 chip travels as
base64 to stay within Triton's byte tensor contract.

Input request JSON:
  s2: str          base64-encoded float32 chip
  shape: [int]     chip shape (e.g. [6, 224, 224] or [1, 6, 1, 224, 224])
  scene_id: str | null
  scene_datetime: str | null
  cloud_cover: float | null

Output result JSON:
  ok, elapsed_s, device, pct_water_within_500m, pct_water_full,
  pred_b64, pred_shape, shape, scene_id, scene_datetime, cloud_cover
"""
import base64
import json
import time

import numpy as np
import triton_python_backend_utils as pb_utils


def _decode_array(b64: str, shape: list, dtype: str = "float32") -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype).reshape(shape)


class TritonPythonModel:
    _model = None
    _device = "cpu"

    def initialize(self, args):
        import importlib.util
        import torch
        from huggingface_hub import hf_hub_download

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        BASE_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
        V2_REPO = "msradam/Prithvi-EO-2.0-NYC-Pluvial"

        from terratorch.cli_tools import LightningInferenceModel

        base_config = hf_hub_download(BASE_REPO, "config.yaml")
        inference_py = hf_hub_download(BASE_REPO, "inference.py")

        v2_yaml = v2_ckpt = None
        for name in ("prithvi_nyc_phase14.yaml", "config.yaml"):
            try:
                v2_yaml = hf_hub_download(V2_REPO, name)
                break
            except Exception:
                continue
        for name in ("prithvi_nyc_pluvial_v2.ckpt", "best_val_loss.ckpt", "model.ckpt"):
            try:
                v2_ckpt = hf_hub_download(V2_REPO, name)
                break
            except Exception:
                continue

        if v2_yaml and v2_ckpt:
            m = LightningInferenceModel.from_config(v2_yaml, v2_ckpt)
            if getattr(getattr(m, "datamodule", None), "test_transform", None) is None:
                import albumentations as A
                from albumentations.pytorch import ToTensorV2

                m.datamodule.test_transform = A.Compose([ToTensorV2()])
                _old = m.datamodule.aug

                class _DictNormalize:
                    def __init__(self, mean, std):
                        self.mean = torch.as_tensor(mean).view(-1, 1, 1).float()
                        self.std = torch.as_tensor(std).view(-1, 1, 1).float()

                    def __call__(self, sample):
                        if isinstance(sample, dict):
                            img = sample["image"]
                            return {**sample, "image": (img - self.mean.to(img.device)) / self.std.to(img.device)}
                        return (sample - self.mean.to(sample.device)) / self.std.to(sample.device)

                m.datamodule.aug = _DictNormalize(_old.means, _old.stds)
        else:
            base_ckpt = hf_hub_download(
                BASE_REPO, "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
            )
            m = LightningInferenceModel.from_config(base_config, base_ckpt)

        m.model.eval()
        if self._device == "cuda" and torch.cuda.is_available():
            m.model.cuda()

        spec = importlib.util.spec_from_file_location("_prithvi_inference", inference_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._model = m
        pb_utils.Logger.log_info("prithvi_pluvial: ready")

    def execute(self, requests):
        import torch

        responses = []
        for request in requests:
            t0 = time.time()
            raw = pb_utils.get_input_tensor_by_name(request, "request").as_numpy()
            payload = json.loads(
                raw.flatten()[0].decode("utf-8")
                if isinstance(raw.flatten()[0], bytes)
                else raw.flatten()[0]
            )

            try:
                chip = _decode_array(payload["s2"], payload["shape"], "float32")
                if chip.ndim == 3:
                    chip = chip[None, :, None, :, :]
                elif chip.ndim == 4:
                    chip = chip[:, :, None, :, :]

                means_t = torch.tensor(
                    [0.107, 0.107, 0.115, 0.265, 0.235, 0.155], dtype=torch.float32
                ).view(1, 6, 1, 1, 1)
                stds_t = torch.tensor(
                    [0.082, 0.075, 0.085, 0.115, 0.11, 0.1], dtype=torch.float32
                ).view(1, 6, 1, 1, 1)

                chip_t = torch.from_numpy(chip).float()
                if self._device == "cuda":
                    chip_t = chip_t.to("cuda")
                    means_t = means_t.to("cuda")
                    stds_t = stds_t.to("cuda")
                x = (chip_t - means_t) / stds_t

                with torch.no_grad():
                    out = self._model.model(x)
                    logits = out.output if hasattr(out, "output") else out

                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype("uint8")
                pct_full = float(100.0 * pred.mean())
                h, w = pred.shape
                yy, xx = np.indices(pred.shape)
                cy, cx = h // 2, w // 2
                dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                mask = dist <= min(50, min(h, w) // 4)
                pct_500m = float(100.0 * pred[mask].mean()) if mask.any() else pct_full
                pred_b64 = base64.b64encode(pred.tobytes()).decode("ascii")

                result = {
                    "ok": True,
                    "elapsed_s": round(time.time() - t0, 3),
                    "device": self._device,
                    "pct_water_within_500m": round(pct_500m, 3),
                    "pct_water_full": round(pct_full, 3),
                    "scene_id": payload.get("scene_id"),
                    "scene_datetime": payload.get("scene_datetime"),
                    "cloud_cover": payload.get("cloud_cover"),
                    "shape": [int(h), int(w)],
                    "pred_b64": pred_b64,
                    "pred_shape": [int(h), int(w)],
                }
            except Exception as e:
                result = {"ok": False, "err": f"{type(e).__name__}: {e}"}

            out_t = pb_utils.Tensor(
                "result",
                np.array([json.dumps(result).encode()], dtype=object),
            )
            responses.append(pb_utils.InferenceResponse([out_t]))
        return responses

    def finalize(self):
        self._model = None
