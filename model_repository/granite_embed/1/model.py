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
        from sentence_transformers import SentenceTransformer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(
            "ibm-granite/granite-embedding-278m-multilingual",
            device=self._device,
            trust_remote_code=True,
        )
        pb_utils.Logger.log_info("granite_embed: ready")

    def execute(self, requests):
        responses = []
        for request in requests:
            t0 = time.time()
            raw = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            texts = [r.decode("utf-8") if isinstance(r, bytes) else r for r in raw.flatten()]

            vecs = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            vecs = np.array(vecs, dtype=np.float32)

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
