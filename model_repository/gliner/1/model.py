"""Triton Python backend — GLiNER medium-v2.1.

Input:  text   BYTES[1]    document text
        labels BYTES[-1]   entity type labels (one per element)
Output: result BYTES[1]    JSON: {ok, elapsed_s, device, entities:[{label,text,start,end,score}]}
"""
import json
import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    _model = None

    def initialize(self, args):
        import torch
        # Newer transformers (4.48+) fetches additional_chat_templates during tokenizer
        # load and 404s on deberta-v3-base (used by GLiNER). Patch at the source module
        # AND the direct-import copy in tokenization_utils_base.
        try:
            import transformers.utils.hub as _hub
            import transformers.tokenization_utils_base as _tok_base

            _orig_lrt = getattr(_hub, "list_repo_templates", None)
            if _orig_lrt is not None:
                def _safe_lrt(*a, **kw):
                    try:
                        return _orig_lrt(*a, **kw)
                    except Exception:
                        return []
                _hub.list_repo_templates = _safe_lrt
                if hasattr(_tok_base, "list_repo_templates"):
                    _tok_base.list_repo_templates = _safe_lrt
        except Exception:
            pass
        from gliner import GLiNER

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        if self._device == "cuda":
            self._model = self._model.to("cuda")
        pb_utils.Logger.log_info("gliner: ready")

    def execute(self, requests):
        responses = []
        for request in requests:
            t0 = time.time()
            text_raw = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()
            labels_raw = pb_utils.get_input_tensor_by_name(request, "labels").as_numpy()

            text = text_raw.flatten()[0]
            if isinstance(text, bytes):
                text = text.decode("utf-8")

            labels = [
                l.decode("utf-8") if isinstance(l, bytes) else l
                for l in labels_raw.flatten()
            ]

            try:
                ents = self._model.predict_entities(text, labels)
                result = {
                    "ok": True,
                    "elapsed_s": round(time.time() - t0, 3),
                    "device": self._device,
                    "entities": [
                        {
                            "label": e["label"],
                            "text": e["text"],
                            "start": int(e.get("start", 0)),
                            "end": int(e.get("end", 0)),
                            "score": float(e.get("score", 0)),
                        }
                        for e in ents
                    ],
                }
            except Exception as e:
                result = {"ok": False, "err": f"{type(e).__name__}: {e}", "entities": []}

            out = pb_utils.Tensor(
                "result",
                np.array([json.dumps(result).encode()], dtype=object),
            )
            responses.append(pb_utils.InferenceResponse([out]))
        return responses

    def finalize(self):
        self._model = None
