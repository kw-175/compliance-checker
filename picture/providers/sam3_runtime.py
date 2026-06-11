from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from picture.domain.exceptions import ProviderNotAvailableError


_runtime_cache: dict[tuple[str, str], dict[str, Any]] = {}
_runtime_lock = threading.Lock()


def get_sam3_runtime(model_dir: str | Path, device: str = "auto") -> dict[str, Any]:
    path = Path(model_dir)
    key = (str(path.resolve()), device)
    with _runtime_lock:
        runtime = _runtime_cache.get(key)
        if runtime is not None:
            return runtime
        if not (path / "sam3.pt").exists() and not (path / "model.safetensors").exists():
            raise ProviderNotAvailableError(f"SAM3 weights at {path}")
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor  # type: ignore[attr-defined]
        except ImportError as exc:
            raise ProviderNotAvailableError("transformers with Sam3Model/Sam3Processor") from exc
        except Exception as exc:
            raise ProviderNotAvailableError("transformers with Sam3Model/Sam3Processor") from exc

        resolved_device = _resolve_device(device, torch)
        model = Sam3Model.from_pretrained(str(path), local_files_only=True).to(resolved_device)
        model.eval()
        processor = Sam3Processor.from_pretrained(str(path), local_files_only=True)
        runtime = {
            "model": model,
            "processor": processor,
            "device": resolved_device,
            "torch": torch,
            "model_dir": str(path),
        }
        _runtime_cache[key] = runtime
        return runtime


def _resolve_device(requested: str, torch: Any) -> str:
    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise ProviderNotAvailableError(f"SAM3 requested device {requested!r}, but CUDA is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    raise ProviderNotAvailableError("SAM3 requires CUDA because GPU-first execution is configured")
