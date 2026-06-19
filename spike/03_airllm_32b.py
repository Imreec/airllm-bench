"""G-SPIKE T2.3 + T2.4 + T2.5 — one Qwen2.5-32B token via AirLLM, timed.

The decisive measurement: seconds-per-token for the 32B model through AirLLM. That number
sizes the whole experiment matrix. Also:
  - cold vs warm (generate 1 token, then again)  -> page-cache signal (T2.4)
  - a few-token run                              -> per-token decode estimate (T2.3)
  - a logits probe on the AirLLM model           -> can we compute perplexity? (T2.5 / risk R5)

WARNING: first run downloads ~tens of GB and each token streams all layers from disk. Minutes,
not seconds. If it hangs for hours, Ctrl-C and paste the last output — we replan, not wait.

Set the shard path to your NVMe first (never the HDD):
  PowerShell:  $env:AIRLLM_SHARDS = "D:\airllm_shards"
Run:
  uv run --extra runtime python spike/03_airllm_32b.py --compression 4bit
Paste the SUMMARY block back.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import psutil
import pynvml
import torch

OUT = Path(__file__).parent / "out"


class PeakSampler:
    """Background peaks for VRAM / RSS / power + min system-available, on one clock."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval = interval_s
        self._stop = threading.Event()
        self.peak_vram_mb = 0.0
        self.peak_rss_mb = 0.0
        self.peak_power_w = 0.0
        self.min_avail_mb = float("inf")
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_power_w = max(
                self.peak_power_w, pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000
            )
            self.peak_vram_mb = max(
                self.peak_vram_mb, pynvml.nvmlDeviceGetMemoryInfo(self._h).used / 1024**2
            )
            self.peak_rss_mb = max(self.peak_rss_mb, self._proc.memory_info().rss / 1024**2)
            self.min_avail_mb = min(self.min_avail_mb, psutil.virtual_memory().available / 1024**2)
            time.sleep(self.interval)

    def __enter__(self) -> PeakSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        pynvml.nvmlShutdown()


def time_generate(model, input_ids, n_tokens: int) -> float:
    t0 = time.perf_counter()
    model.generate(input_ids, max_new_tokens=n_tokens, use_cache=True, return_dict_in_generate=True)
    return time.perf_counter() - t0


def probe_logits(model, input_ids) -> str:
    """Does the AirLLM model yield logits for a forward pass? (perplexity / R5)."""
    try:
        with torch.no_grad():
            out = model(input_ids)
        logits = getattr(out, "logits", out)
        return f"OK shape={tuple(logits.shape)}"
    except Exception as exc:  # noqa: BLE001
        return f"FAILED {type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--compression", choices=["none", "4bit", "8bit"], default="4bit")
    ap.add_argument("--prompt", default="What is virtual memory?")
    ap.add_argument("--decode-tokens", type=int, default=3)
    args = ap.parse_args()

    shards = os.environ.get("AIRLLM_SHARDS")
    if not shards:
        print(
            "WARNING: AIRLLM_SHARDS not set — AirLLM uses its default cache (verify it's the NVMe!)."
        )
    compression = None if args.compression == "none" else args.compression
    if shards:
        Path(shards).mkdir(parents=True, exist_ok=True)  # AirLLM check_space needs it to exist

    from airllm import AutoModel

    print(f"Loading {args.model}  compression={args.compression}  shards={shards or '<default>'}")
    print("(first run downloads + shards the model — this is the slow part)")
    t_load0 = time.perf_counter()
    kwargs = {"compression": compression} if compression else {}
    if shards:
        kwargs["layer_shards_saving_path"] = shards
    model = AutoModel.from_pretrained(args.model, **kwargs)
    load_s = time.perf_counter() - t_load0

    tokens = model.tokenizer(
        [args.prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        padding=False,
    )
    input_ids = tokens["input_ids"].cuda()

    with PeakSampler() as s:
        print("Generating 1 token (COLD)...")
        cold_1 = time_generate(model, input_ids, 1)
        print(f"  cold 1-token: {cold_1:.1f} s")
        print("Generating 1 token (WARM)...")
        warm_1 = time_generate(model, input_ids, 1)
        print(f"  warm 1-token: {warm_1:.1f} s")
        print(f"Generating {args.decode_tokens} tokens (decode rate)...")
        multi = time_generate(model, input_ids, args.decode_tokens)
        logits = probe_logits(model, input_ids)

    res = {
        "model": args.model,
        "compression": args.compression,
        "load_s": round(load_s, 1),
        "cold_1tok_s": round(cold_1, 1),
        "warm_1tok_s": round(warm_1, 1),
        "cold_over_warm": round(cold_1 / warm_1, 2) if warm_1 else None,
        "decode_tokens": args.decode_tokens,
        "decode_total_s": round(multi, 1),
        "approx_s_per_token": round(multi / max(args.decode_tokens, 1), 1),
        "peak_vram_mb": round(s.peak_vram_mb),
        "peak_rss_mb": round(s.peak_rss_mb),
        "min_sys_available_mb": round(s.min_avail_mb),
        "peak_gpu_power_w": round(s.peak_power_w, 1),
        "logits_probe": logits,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"airllm_32b_{args.compression}.json").write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 60)
    print("SUMMARY (T2.3 + T2.4 + T2.5) — paste this back")
    print("=" * 60)
    for k, v in res.items():
        print(f"  {k:>22}: {v}")
    print("=" * 60)
    print("cold/warm ratio >> 1 → page cache helps (model fits RAM); ~1 → it doesn't.")


if __name__ == "__main__":
    main()
