#!/usr/bin/env python
"""Launch one vLLM server per GPU, with settings that actually work on Turing.

Encodes the things that cost real debugging time on a T4 (sm75):

* **fp16, never bf16.** ``torch.cuda.is_bf16_supported()`` returns True on a T4,
  but there is no native bf16 path - it is emulated and slow.
* **Attention backend must be chosen explicitly.** vLLM auto-selects
  FlexAttention on sm75 because FlashAttention-2 needs sm80+. FlexAttention is a
  ``torch.compile`` fallback, not a real kernel: measured ~22 tok/s aggregate
  across 4 concurrent requests. TRITON_ATTN is a genuine kernel and is the right
  default here.
* **One server per GPU beats tensor-parallel.** The two T4s are connected by
  PCIe host bridge (``PHB``), with no NVLink, so TP=2 spends its time moving
  activations. Two independent replicas use both cards with no interconnect
  cost - and an agent workload is embarrassingly parallel across episodes.
* **Context is bounded by KV cache, not by the model.** 32k needs 4.50 GiB of
  KV cache against ~4.07 GiB free after fp16 weights, so the engine refuses to
  start. 24576 fits.

Usage::

    python scripts/serve.py --gpus 0 1 --model Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Attention backends that have real kernels on each capability tier. On sm80+
# we let vLLM choose (FlashAttention). Below that, auto-selection lands on
# FlexAttention, which works but is roughly an order of magnitude too slow.
_SM75_BACKEND = "TRITON_ATTN"


def gpu_capability(index: int = 0) -> tuple[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    major, minor = out[index].strip().split(".")
    return int(major), int(minor)


def build_command(args: argparse.Namespace, gpu: int, port: int) -> str:
    cap = gpu_capability(gpu if gpu < 8 else 0)
    dtype = args.dtype or ("float16" if cap < (8, 0) else "bfloat16")
    backend = args.attention_backend or (_SM75_BACKEND if cap < (8, 0) else "")
    env = f"CUDA_VISIBLE_DEVICES={gpu}"
    if backend:
        env += f" VLLM_ATTENTION_BACKEND={backend}"
    return (
        f"{env} python -m vllm.entrypoints.openai.api_server"
        f" --model {args.model} --served-model-name {args.served_name}"
        f" --dtype {dtype} --max-model-len {args.max_model_len}"
        f" --gpu-memory-utilization {args.gpu_memory_utilization}"
        f" --enable-prefix-caching"
        f" --enable-auto-tool-choice --tool-call-parser {args.tool_parser}"
        f" --max-num-seqs {args.max_num_seqs}"
        f" --port {port} --host 127.0.0.1"
    )


def wait_until_up(port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=3
            ) as response:
                json.load(response)
                return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            time.sleep(5)
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--served-name", default="qwen3-4b")
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--base-port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=24576)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.93)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--tool-parser", default="hermes")
    p.add_argument("--dtype", default=None, help="default: fp16 below sm80")
    p.add_argument("--attention-backend", default=None)
    p.add_argument("--log-dir", default="/content/logs")
    p.add_argument("--wait", type=float, default=600.0, help="0 to return immediately")
    args = p.parse_args(argv)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ports = []

    for offset, gpu in enumerate(args.gpus):
        port = args.base_port + offset
        ports.append(port)
        command = build_command(args, gpu, port)
        script = log_dir / f"serve_{port}.sh"
        script.write_text(f"{command} 2>&1\necho ___SERVER_EXITED___ rc=$?\n")
        log = log_dir / f"vllm_{port}.log"
        # Detached, because a notebook cell that blocks on a server is a cell
        # that can never be interrupted through the Colab MCP bridge.
        subprocess.Popen(
            f"nohup bash {script} > {log} 2>&1 &", shell=True
        )
        print(f"gpu {gpu} -> port {port}  (log: {log})")
        print(f"  {command}")

    if args.wait <= 0:
        return 0

    ok = True
    for port in ports:
        if wait_until_up(port, args.wait):
            print(f"port {port}: UP")
        else:
            ok = False
            print(f"port {port}: FAILED to come up in {args.wait:.0f}s; see log")
    print("base urls:", " ".join(f"http://127.0.0.1:{p}/v1" for p in ports))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
