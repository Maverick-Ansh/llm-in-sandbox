# Resume from here

State as of 2026-08-11. Everything is resumable — no work is lost if the Colab
runtime died.

## Where things stand

Run directory: `runs/pilot_qwen3-4b/` on the Colab runtime (not in git).

| arm | n | status |
|---|---:|---|
| `direct` | 100 | **complete**, graded |
| `sandbox` | 100 | **complete**, graded — results in [`../results/qwen3-4b-mmlu-pro.md`](../results/qwen3-4b-mmlu-pro.md) |
| `sandbox_neutral` | 100 | **launched, in progress** — the separating experiment |

Headline so far: sandbox −8.0 pp overall (p=0.096), maths **−32 pp** (p=0.008),
tokens 0.84× overall but 2.75× on biomedicine.

## Finishing the `sandbox_neutral` arm

It answers the one question the first result cannot: is the maths collapse the
*environment* or the *prompt*? `sandbox_neutral` is byte-identical to `sandbox`
except that it stops telling the model not to reason.

```bash
# 1. If the runtime died, serve again first
python scripts/serve.py --model Qwen/Qwen3-4B-Instruct-2507 \
    --served-name qwen3-4b --gpus 0 1 --max-model-len 24576 --wait 900

# 2. Resume (skips the 200 completed episodes, runs only what is missing)
python scripts/run_sweep.py \
    --benchmark mmlu_pro --n 25 --model qwen3-4b \
    --base-url http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 \
    --modes sandbox_neutral --max-turns 20 --max-tokens-per-turn 2048 \
    --workers 8 --out runs/pilot_qwen3-4b

# 3. ONLY AFTER the sweep exits. regrade rewrites results.jsonl while the sweep
#    appends to it, so running them together loses rows.
python scripts/regrade.py runs/pilot_qwen3-4b

# 4. Both comparisons
python scripts/report.py runs/pilot_qwen3-4b/results.jsonl \
    --treatment sandbox/full         --out runs/pilot_qwen3-4b/report_sandbox.md
python scripts/report.py runs/pilot_qwen3-4b/results.jsonl \
    --treatment sandbox_neutral/full --out runs/pilot_qwen3-4b/report_neutral.md
```

## How to read the answer

The comparison that matters is **maths accuracy**, `sandbox_neutral` vs the
88.0% baseline (`sandbox` scored 56.0%):

- **Recovers toward ~88%** → the effect was the *prompt*. "Compute instead of
  reasoning" is a bad instruction at 4B, and the paper's negative small-model
  result may be substantially a prompt-design artefact rather than evidence
  about computer environments.
- **Stays near 56%** → the effect is *environmental*. Multi-turn tool-calling
  itself disrupts this model's reasoning, which is a genuine reproduction of the
  paper's finding and a stronger claim than the paper makes.
- **Lands in between** → both contribute; report the split and do not round it
  to whichever story is tidier.

Also worth checking: the **zero-command rate**. It was 46/100 under the original
prompt. If the neutral prompt lowers it *and* maths recovers, the mechanism is
that the original prompt suppressed reasoning without reliably buying
computation — which is the hypothesis in `results/qwen3-4b-mmlu-pro.md`.

## Standing traps

1. **`regrade.py` rewrites `results.jsonl`; the sweep appends to it.** Never run
   them concurrently. `pgrep -f run_sweep.py` self-matches inside `sh -c`, so a
   count of 1 usually means "not running" — confirm with `___SWEEP_DONE___` in
   the log instead.
2. **Grade both arms with one extractor version.** Step 3 is not optional; skip
   it and part of any measured difference is the code change between runs.
3. **n=100 resolves ~10 pp.** A simulated genuine +8 pp effect reads as +13.3 pp
   with CI [−3.3, +30.0], p=0.185. Only the maths effect is large enough to
   resolve; treat other rows as noise.

## Known-good environment

- vLLM 0.11.0, `transformers==4.56.2` (5.x breaks it), torch 2.8.0+cu128
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` — default FlexAttention on sm75 is ~5x slower
- `--max-model-len 24576` — 32k does not fit in KV cache beside fp16 weights
- `unshare` denied on Colab; the seccomp backend covers the network ablation
