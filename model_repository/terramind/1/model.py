"""Triton Python backend — TerraMind NYC adapters (lulc / buildings / synthesis).

All three adapters live in one backend instance to share the heavy
backbone weights that all three load. Dispatch is via payload.adapter.

Input request JSON:
  adapter: str        "lulc" | "buildings" | "synthesis"
  s2: str | null      base64 float32 chip (lulc/buildings)
  s2_shape: [int]
  s1: str | null      base64 float32 chip (optional)
  s1_shape: [int]
  dem: str | null     base64 float32 chip (synthesis)
  dem_shape: [int]

Output result JSON: mirrors riprap-models /v1/terramind response contract.
"""
import base64
import json
import time
from threading import Lock

import numpy as np
import triton_python_backend_utils as pb_utils

_TERRAMIND_REPO = "msradam/TerraMind-NYC-Adapters"
_TERRAMIND_SPECS = {
    "lulc":      {"subdir": "lulc_nyc",      "num_classes": 5,
                   "labels": ["Trees", "Cropland", "Built", "Bare", "Water"]},
    "buildings": {"subdir": "buildings_nyc", "num_classes": 2,
                   "labels": ["Background", "Building"]},
    "synthesis": {"subdir": None, "num_classes": None,
                   "labels": ["Water", "Trees", "Grass", "Flooded vegetation",
                              "Crops", "Scrub/Shrub", "Built", "Bare ground",
                              "Snow/Ice", "Clouds"]},
}
_SYNTH_TIMESTEPS = 10


def _decode_array(b64: str, shape: list, dtype: str = "float32") -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype).reshape(shape)


def _build_chip_tensor(np_arr, n_timesteps: int = 4):
    import torch
    t = torch.from_numpy(np_arr).float()
    if t.ndim == 5:
        return t
    if t.ndim == 4:
        return t.unsqueeze(0)
    if t.ndim == 3:
        t = t.unsqueeze(1)
        if t.shape[1] == 1:
            t = t.repeat(1, n_timesteps, 1, 1)
        return t.unsqueeze(0)
    raise ValueError(f"unexpected chip shape {tuple(t.shape)}")


class TritonPythonModel:
    _instances: dict = {}
    _lock = Lock()
    _device = "cpu"
    _synth_timesteps = _SYNTH_TIMESTEPS

    def initialize(self, args):
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._synth_timesteps = int(
            args.get("model_config", {})
            .get("parameters", {})
            .get("SYNTH_TIMESTEPS", {})
            .get("string_value", str(_SYNTH_TIMESTEPS))
        )
        # Pre-load all adapters.
        for adapter in ("lulc", "buildings", "synthesis"):
            try:
                self._get_model(adapter)
            except Exception as e:
                pb_utils.Logger.log_error(f"terramind/{adapter} preload failed: {e}")
        pb_utils.Logger.log_info("terramind: adapters ready")

    def _get_model(self, adapter: str):
        key = f"terramind_{adapter}"
        if key in self._instances:
            return self._instances[key]
        with self._lock:
            if key in self._instances:
                return self._instances[key]
            import torch
            if adapter == "synthesis":
                import terratorch.models.backbones.terramind.model.terramind_register  # noqa
                from terratorch.registry import FULL_MODEL_REGISTRY
                m = FULL_MODEL_REGISTRY.build(
                    "terratorch_terramind_v1_base_generate",
                    modalities=["DEM"],
                    output_modalities=["LULC"],
                    pretrained=True,
                    timesteps=self._synth_timesteps,
                )
                if self._device == "cuda" and torch.cuda.is_available():
                    m = m.to("cuda")
                m.eval()
            else:
                from huggingface_hub import snapshot_download
                from peft import LoraConfig, inject_adapter_in_model
                from safetensors.torch import load_file
                from terratorch.tasks import SemanticSegmentationTask

                spec = _TERRAMIND_SPECS[adapter]
                adapter_root = snapshot_download(
                    _TERRAMIND_REPO, allow_patterns=[f"{spec['subdir']}/*"]
                )
                task = SemanticSegmentationTask(
                    model_factory="EncoderDecoderFactory",
                    model_args=dict(
                        backbone="terramind_v1_base",
                        backbone_pretrained=True,
                        backbone_modalities=["S2L2A", "S1RTC", "DEM"],
                        backbone_use_temporal=True,
                        backbone_temporal_pooling="concat",
                        backbone_temporal_n_timestamps=4,
                        necks=[
                            {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                            {"name": "LearnedInterpolateToPyramidal"},
                        ],
                        decoder="UNetDecoder",
                        decoder_channels=[512, 256, 128, 64],
                        head_dropout=0.1,
                        num_classes=spec["num_classes"],
                    ),
                    loss="ce", lr=1e-4, freeze_backbone=False, freeze_decoder=False,
                )
                inject_adapter_in_model(LoraConfig(
                    r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=["attn.qkv", "attn.proj"], bias="none",
                ), task.model.encoder)
                adapter_dir = f"{adapter_root}/{spec['subdir']}"
                lora = load_file(f"{adapter_dir}/adapter_model.safetensors")
                head = load_file(f"{adapter_dir}/decoder_head.safetensors")
                task.model.encoder.load_state_dict(
                    {k.removeprefix("encoder."): v for k, v in lora.items()
                     if k.startswith("encoder.")}, strict=False
                )
                for sub in ("decoder", "neck", "head", "aux_heads"):
                    ss = {k[len(sub) + 1:]: v for k, v in head.items()
                          if k.startswith(sub + ".")}
                    if ss and hasattr(task.model, sub):
                        getattr(task.model, sub).load_state_dict(ss, strict=False)
                if self._device == "cuda" and torch.cuda.is_available():
                    task = task.to("cuda")
                task.eval()
                m = task
            self._instances[key] = m
            pb_utils.Logger.log_info(f"terramind/{adapter}: loaded")
            return m

    def _run_synthesis(self, payload: dict) -> dict:
        import torch
        t0 = time.time()
        if not payload.get("dem") or not payload.get("dem_shape"):
            return {"ok": False, "err": "synthesis requires dem + dem_shape"}
        model = self._get_model("synthesis")
        dem_np = _decode_array(payload["dem"], payload["dem_shape"])
        dem_t = torch.from_numpy(dem_np).float()
        if dem_t.ndim == 2:
            dem_t = dem_t.unsqueeze(0).unsqueeze(0)
        elif dem_t.ndim == 3:
            dem_t = dem_t.unsqueeze(0)
        if self._device == "cuda":
            dem_t = dem_t.to("cuda")
        spec = _TERRAMIND_SPECS["synthesis"]
        with torch.no_grad():
            out = model({"DEM": dem_t}, timesteps=self._synth_timesteps, verbose=False)
        lulc = out["LULC"]
        if hasattr(lulc, "detach"):
            lulc = lulc.detach().cpu().numpy()
        if lulc.ndim == 4:
            lulc = lulc[0]
        class_idx = lulc.argmax(axis=0)
        unique, counts = np.unique(class_idx, return_counts=True)
        total = float(class_idx.size) or 1.0
        fractions = {}
        for u, c in zip(unique, counts):
            u = int(u)
            label = spec["labels"][u] if 0 <= u < len(spec["labels"]) else f"class_{u}"
            fractions[label] = round(100.0 * c / total, 2)
        fractions = dict(sorted(fractions.items(), key=lambda kv: kv[1], reverse=True))
        dom = next(iter(fractions)) if fractions else "unknown"
        pred_b64 = base64.b64encode(class_idx.astype("uint8").tobytes()).decode("ascii")
        return {
            "ok": True, "adapter": "synthesis",
            "elapsed_s": round(time.time() - t0, 3), "device": self._device,
            "synthetic_modality": True, "tim_chain": ["DEM", "LULC_synthetic"],
            "diffusion_steps": self._synth_timesteps,
            "class_fractions": fractions, "dominant_class": dom,
            "dominant_pct": fractions.get(dom, 0.0),
            "n_classes_observed": len(fractions),
            "shape": list(lulc.shape), "n_pixels": int(class_idx.size),
            "pred_b64": pred_b64, "pred_shape": [int(s) for s in class_idx.shape],
            "class_labels": spec["labels"],
            "label_schema": "ESRI 2020-2022 Land Cover",
        }

    def _run_adapter(self, adapter: str, payload: dict) -> dict:
        import torch
        t0 = time.time()
        spec = _TERRAMIND_SPECS[adapter]
        if not payload.get("s2") or not payload.get("s2_shape"):
            return {"ok": False, "err": f"{adapter} requires s2 + s2_shape"}
        task = self._get_model(adapter)
        s2 = _decode_array(payload["s2"], payload["s2_shape"])
        chips = {"S2L2A": _build_chip_tensor(s2)}
        if payload.get("s1") and payload.get("s1_shape"):
            chips["S1RTC"] = _build_chip_tensor(_decode_array(payload["s1"], payload["s1_shape"]))
        if payload.get("dem") and payload.get("dem_shape"):
            chips["DEM"] = _build_chip_tensor(_decode_array(payload["dem"], payload["dem_shape"]))
        if self._device == "cuda":
            chips = {k: v.to("cuda") for k, v in chips.items()}

        def _fwd(x):
            out = task.model(x)
            return out.output if hasattr(out, "output") else out

        h_chip, w_chip = int(chips["S2L2A"].shape[-2]), int(chips["S2L2A"].shape[-1])
        with torch.no_grad():
            if h_chip == 224 and w_chip == 224:
                logits = _fwd(chips)
            else:
                from terratorch.tasks.tiled_inference import tiled_inference
                logits = tiled_inference(
                    lambda x, **_: _fwd(x), chips,
                    out_channels=spec["num_classes"],
                    h_crop=224, w_crop=224, h_stride=128, w_stride=128,
                    average_patches=True, blend_overlaps=True, padding="reflect",
                )
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype("uint8")
        n = max(int(pred.size), 1)
        fractions = {
            spec["labels"][i]: round(100.0 * float((pred == i).sum()) / n, 2)
            for i in range(spec["num_classes"])
        }
        fractions = {k: v for k, v in fractions.items() if v > 0}
        dom_idx = int(max(range(spec["num_classes"]), key=lambda i: int((pred == i).sum())))
        n_components = None
        if adapter == "buildings":
            try:
                from scipy.ndimage import label
                _, n_components = label((pred == 1).astype("uint8"))
                n_components = int(n_components)
            except Exception:
                pass
        pred_b64 = base64.b64encode(pred.tobytes()).decode("ascii")
        return {
            "ok": True, "adapter": adapter,
            "elapsed_s": round(time.time() - t0, 3), "device": self._device,
            "shape": list(pred.shape), "n_pixels": int(pred.size),
            "class_fractions": fractions,
            "dominant_class": spec["labels"][dom_idx],
            "dominant_pct": fractions.get(spec["labels"][dom_idx], 0.0),
            "pct_buildings": round(100.0 * float((pred == 1).sum()) / n, 2)
                             if adapter == "buildings" else None,
            "n_building_components": n_components,
            "pred_b64": pred_b64, "pred_shape": [int(s) for s in pred.shape],
            "class_labels": spec["labels"],
        }

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "request").as_numpy()
            payload = json.loads(
                raw.flatten()[0].decode("utf-8")
                if isinstance(raw.flatten()[0], bytes)
                else raw.flatten()[0]
            )
            adapter = payload.get("adapter", "lulc")
            try:
                if adapter == "synthesis":
                    result = self._run_synthesis(payload)
                elif adapter in _TERRAMIND_SPECS:
                    result = self._run_adapter(adapter, payload)
                else:
                    result = {"ok": False, "err": f"unknown adapter {adapter!r}"}
            except Exception as e:
                result = {"ok": False, "adapter": adapter, "err": f"{type(e).__name__}: {e}"}

            out_t = pb_utils.Tensor(
                "result",
                np.array([json.dumps(result).encode()], dtype=object),
            )
            responses.append(pb_utils.InferenceResponse([out_t]))
        return responses

    def finalize(self):
        self._instances.clear()
