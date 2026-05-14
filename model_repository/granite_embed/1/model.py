"""Triton Python backend — Granite Embedding 278M.

Input:  texts  BYTES[-1]           one string per element
Output: vectors FP32[-1, 768]      L2-normalised embeddings
        meta    BYTES[1]           JSON: {ok, elapsed_s, n, dim, device}
"""
import json
import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    _model = None

    def initialize(self, args):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "ibm-granite/granite-embedding-278m-multilingual"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()
        pb_utils.Logger.log_info("granite_embed: ready")

    def _encode(self, texts):
        import torch
        import torch.nn.functional as F
        encoded = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**encoded)
        # Mean pool over token dimension, then L2-normalise
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vecs = F.normalize(vecs, p=2, dim=1)
        return vecs.cpu().numpy().astype(np.float32)

    def execute(self, requests):
        responses = []
        for request in requests:
            t0 = time.time()
            raw = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            texts = [r.decode("utf-8") if isinstance(r, bytes) else r for r in raw.flatten()]

            vecs = self._encode(texts)

            meta = json.dumps({
                "ok": True,
                "elapsed_s": round(time.time() - t0, 3),
                "n": len(texts),
                "dim": vecs.shape[-1],
                "device": self._device,
            })

            out_vecs = pb_utils.Tensor("vectors", vecs)
            out_meta = pb_utils.Tensor(
                "meta", np.array([meta.encode()], dtype=object)
            )
            responses.append(pb_utils.InferenceResponse([out_vecs, out_meta]))
        return responses

    def finalize(self):
        self._model = None
