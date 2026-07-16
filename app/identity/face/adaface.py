from __future__ import annotations

import threading

import numpy as np

from ...config import settings

_lock = threading.Lock()
_state: dict = {"ready": False, "model": None, "error": None}


def _ensure_model():
    if _state["ready"] or _state["error"] is not None:
        return _state["model"]
    with _lock:
        if _state["ready"] or _state["error"] is not None:
            return _state["model"]
        try:
            import sys

            import torch

            root = settings.face_adaface_root
            if root not in sys.path:
                sys.path.insert(0, root)
            import net as adaface_net

            model = adaface_net.build_model(settings.face_adaface_arch)
            state_dict = torch.load(settings.face_adaface_weights, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if any(key.startswith("net.") for key in state_dict):
                state_dict = {key[4:]: value for key, value in state_dict.items() if key.startswith("net.")}
            elif any(key.startswith("model.") for key in state_dict):
                state_dict = {key[6:]: value for key, value in state_dict.items() if key.startswith("model.")}
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            configured = (settings.face_device or "auto").strip().lower()
            use_cuda = configured != "cpu" and torch.cuda.is_available()
            device = "cuda" if use_cuda else "cpu"
            model = model.to(device)
            _state.update({"torch": torch, "model": model, "device": device, "ready": True})
        except Exception as exc:  # noqa: BLE001
            _state["error"] = f"{type(exc).__name__}: {exc}"
        return _state["model"]


def embed(bgr_face: np.ndarray) -> np.ndarray | None:
    model = _ensure_model()
    if model is None:
        return None
    try:
        import cv2

        torch = _state["torch"]
        bgr = cv2.resize(bgr_face, (112, 112)) if bgr_face.shape[:2] != (112, 112) else bgr_face
        normalized = ((bgr.astype(np.float32) / 255.0) - 0.5) / 0.5
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)[None]).float().to(_state["device"])
        with torch.no_grad():
            output = model(tensor)
        feature = (output[0] if isinstance(output, (tuple, list)) else output)
        feature = feature.reshape(-1).cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(feature))
        return feature / norm if norm > 0 else feature
    except Exception as exc:  # noqa: BLE001
        _state.setdefault("embed_error", str(exc))
        return None


def load_error() -> str | None:
    return _state.get("error")


__all__ = ["embed", "load_error"]
