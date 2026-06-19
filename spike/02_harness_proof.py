"""G-SPIKE T2.2 + T2.5 — prove the instrumentation end-to-end on a tiny model.

This is the *pattern* the real harness will reuse, validated on a model that finishes in
seconds:
  - streaming per-token timestamps  -> TTFT, ITL series, TPOT, throughput (NOT total/tokens)
  - a background sampler            -> NVML power+VRAM, psutil RSS + system memory, on one clock
  - a logits probe                  -> confirms perplexity is computable

Windows note: psutil exposes no system `cached`/standby figure, so we record `available` and
rely on cold-vs-warm timing (script 03) as the page-cache signal. Surfaced here on purpose.

Run: uv run --extra runtime python spike/02_harness_proof.py
Paste the SUMMARY block back.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import psutil
import pynvml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

OUT = Path(__file__).parent / "out"


class Sampler:
    """Background thread sampling GPU power/VRAM + RAM on one perf_counter clock."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval = interval_s
        self._stop = threading.Event()
        self._samples: list[tuple[float, float, float, float, float, float]] = []
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()
            power_w = pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
            vram_mb = pynvml.nvmlDeviceGetMemoryInfo(self._h).used / 1024**2
            vm = psutil.virtual_memory()
            self._samples.append(
                (
                    t,
                    power_w,
                    vram_mb,
                    self._proc.memory_info().rss / 1024**2,
                    vm.used / 1024**2,
                    vm.available / 1024**2,
                )
            )
            time.sleep(self.interval)

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        pynvml.nvmlShutdown()

    def report(self) -> dict[str, float]:
        s = self._samples
        if len(s) < 2:
            return {}
        energy_j = sum(
            (s[i][1] + s[i - 1][1]) / 2 * (s[i][0] - s[i - 1][0]) for i in range(1, len(s))
        )
        return {
            "peak_gpu_power_w": max(x[1] for x in s),
            "gpu_energy_j": energy_j,
            "peak_vram_mb": max(x[2] for x in s),
            "peak_rss_mb": max(x[3] for x in s),
            "peak_sys_used_mb": max(x[4] for x in s),
            "min_sys_available_mb": min(x[5] for x in s),
            "samples": len(s),
        }


def stream_generate(model, tok, prompt: str, max_new_tokens: int) -> dict[str, float]:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "streamer": streamer,
    }
    t0 = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    stamps = [time.perf_counter() for _ in streamer]
    thread.join()
    if not stamps:
        return {"error": "no tokens streamed"}
    itl = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    return {
        "prompt_tokens": int(input_ids.shape[1]),
        "out_tokens": len(stamps),
        "ttft_s": stamps[0] - t0,
        "tpot_s": (sum(itl) / len(itl)) if itl else float("nan"),
        "throughput_tok_s": len(stamps) / (stamps[-1] - t0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--prompt", default="In one sentence, what is virtual memory?")
    args = ap.parse_args()

    print(f"Loading {args.model} (fp16, cuda)...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")

    with Sampler() as sampler:
        gen = stream_generate(model, tok, args.prompt, args.tokens)

    # Logits probe (T2.5) — confirms perplexity is computable.
    with torch.no_grad():
        ids = tok(args.prompt, return_tensors="pt").to(model.device)
        logits = model(**ids).logits
    logits_shape = tuple(logits.shape)

    res = {"model": args.model, **gen, "logits_shape": logits_shape, **sampler.report()}
    OUT.mkdir(exist_ok=True)
    (OUT / "harness_proof.json").write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 60)
    print("SUMMARY (T2.2 + T2.5) — paste this back")
    print("=" * 60)
    for k, v in res.items():
        print(f"  {k:>22}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
