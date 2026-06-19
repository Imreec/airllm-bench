"""G-SPIKE T2.1 — does the stack import and run on this Windows box?

Checks, independently (one failure doesn't abort the rest):
  - torch sees CUDA + the RTX 3080 Ti, reports VRAM
  - transformers / accelerate / airllm import
  - bitsandbytes does a real 4-bit forward on the GPU (AirLLM's quant backend)

Run: uv run --extra runtime python spike/01_stack_check.py
Paste the whole printed block back.
"""

from __future__ import annotations


def _line(label: str, ok: bool, detail: str) -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label:<16} {detail}")


def check_torch() -> bool:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _line("torch", False, f"import failed: {exc}")
        return False
    cuda = torch.cuda.is_available()
    _line("torch", True, f"v{torch.__version__}  cuda_available={cuda}")
    if cuda:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        _line("gpu", True, f"{props.name}  {vram_gb:.1f} GB VRAM  cc={props.major}.{props.minor}")
    else:
        _line("gpu", False, "torch.cuda.is_available() is False — driver/CUDA build problem")
    return cuda


def check_import(name: str) -> bool:
    try:
        mod = __import__(name)
    except Exception as exc:  # noqa: BLE001
        _line(name, False, f"import failed: {exc}")
        return False
    _line(name, True, f"v{getattr(mod, '__version__', '?')}")
    return True


def check_bitsandbytes_4bit() -> bool:
    try:
        import bitsandbytes as bnb
        import torch

        _line("bitsandbytes", True, f"v{bnb.__version__}")
        with torch.no_grad():
            lin = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float16).to("cuda")
            x = torch.randn(2, 64, dtype=torch.float16, device="cuda")
            y = lin(x)
        _line("bnb 4bit fwd", True, f"output shape {tuple(y.shape)} — 4-bit GPU path works")
        return True
    except Exception as exc:  # noqa: BLE001
        _line("bnb 4bit fwd", False, f"{type(exc).__name__}: {exc}")
        return False


def main() -> None:
    print("=" * 64)
    print("G-SPIKE 01 — stack check (native Windows)")
    print("=" * 64)
    cuda = check_torch()
    imports = {name: check_import(name) for name in ("transformers", "accelerate", "airllm")}
    bnb_ok = check_bitsandbytes_4bit()
    print("-" * 64)
    verdict = "GO" if (cuda and bnb_ok and imports["airllm"]) else "NO-GO (see FAILs above)"
    print(f"  T2.1 verdict: {verdict}")
    print("=" * 64)


if __name__ == "__main__":
    main()
