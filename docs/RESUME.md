# Resume from here

State as of the last session (2026-08-11). Everything below is resumable — no
work is lost if the Colab runtime died.

## Where the run got to

`runs/pilot_qwen3-4b/` on Colab (not in git — regenerate or re-run):

- **`direct` arm: 100/100 episodes complete.** Valid.
  - 17 of them are still marked `no_answer_parsed`. They are *not* failures —
    the model answered tersely (`C`) and the old extractor rejected it. The fix
    is in; those rows need **regrading**, not re-running.
  - 7 previously errored on the request deadline; those were re-queued and were
    running when the session ended.
- **`sandbox` arm: 0/100.** The 42 episodes from the first attempt were
  **deliberately deleted**, along with their trajectories: they ran under the
  tool-argument parser bug, so every `bash` call failed and they exhausted
  their turn budget. Keeping them would have put harness artefacts in the
  results table.

Backups of the pre-edit results file are at
`runs/pilot_qwen3-4b/results.prefix.*.jsonl.bak`.

## To finish it

```bash
# 1. Serve (T4: fp16, TRITON_ATTN, one replica per GPU)
python scripts/serve.py --model Qwen/Qwen3-4B-Instruct-2507 \
    --served-name qwen3-4b --gpus 0 1 --max-model-len 24576 --wait 900

# 2. Resume the sweep. Completed episodes are skipped; errored ones are retried.
python scripts/run_sweep.py \
    --benchmark mmlu_pro --n 25 --model qwen3-4b \
    --base-url http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 \
    --modes direct sandbox --max-turns 20 --max-tokens-per-turn 2048 \
    --workers 8 --out runs/pilot_qwen3-4b

# 3. ONLY AFTER the sweep exits — regrade rewrites results.jsonl while the
#    sweep appends to it, so running them together loses rows.
python scripts/regrade.py runs/pilot_qwen3-4b

# 4. Report
python scripts/report.py runs/pilot_qwen3-4b/results.jsonl \
    --out runs/pilot_qwen3-4b/report.md
```

Step 3 is not optional. The two arms must be graded by one version of the
extractor, or part of any measured difference is the code change between them.

## What to check before believing the numbers

The sandbox arm has never yet run under correct code, so these are unverified:

1. **Do `bash` calls actually succeed?** Check `sandbox_stats.commands > 0` and
   that observations are command output rather than `ERROR: 'command' is
   required`. This is the bug that made the first attempt worthless.
2. **What fraction end in `finish`?** A high `max_turns` or `stalled` share
   means the turn budget is the binding constraint, not the model, and the
   accuracy delta has to be read in that light.
3. **Is the delta concentrated by domain?** The sandbox prompt tells the model
   to *compute* answers by writing scripts. On MMLU-Pro **biomedicine**
   ("which hormone regulates X") that is irrelevant and the turn overhead is
   pure cost, whereas on math/physics it should pay. A negative overall delta
   driven entirely by biomedicine is a different finding from a uniform one.

## Expect a null result, and report it as one

100 paired items resolves roughly a 10+ point swing. A simulation with a
genuine +8pp effect produced an observed +13.3pp with a 95% CI of
`[-3.3, +30.0]` and McNemar p=0.185 — not significant. `report.py` prints CIs
and exact p-values precisely so a null reads as "no detectable effect at this
sample size" rather than becoming an accidental claim in either direction.

To actually detect the paper's ~5-15pp effects, `--n` needs to be several
hundred per domain, which is many hours on a T4.

## Known-good environment

- vLLM 0.11.0, `transformers==4.56.2` (5.x breaks it), torch 2.8.0+cu128
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` — the default FlexAttention on sm75 is
  ~5x slower
- `--max-model-len 24576` — 32k does not fit in KV cache alongside fp16 weights
- `unshare` is denied on Colab; the seccomp backend covers the network ablation
