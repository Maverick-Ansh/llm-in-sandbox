# Qwen3-4B-Instruct-2507 on MMLU-Pro — sandbox vs direct CoT

100 paired items (25 per domain), 2× Tesla T4, vLLM 0.11.0, fp16, temperature 0.
Both arms run over identical items and are graded by one deterministic grader.

Reproduce with `scripts/run_sweep.py` → `scripts/regrade.py` → `scripts/report.py`
(see [`docs/RESUME.md`](../docs/RESUME.md)).

## Accuracy

| domain | n | direct | sandbox | delta (pp) | 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| biomedicine | 25 | 68.0 | 64.0 | −4.0 | [−24.0, +16.0] | 1.000 |
| chemistry | 25 | 64.0 | 64.0 | 0.0 | [0.0, 0.0] | 1.000 |
| math | 25 | 88.0 | 56.0 | **−32.0** | [−52.0, −16.0] | **0.008** |
| physics | 25 | 68.0 | 72.0 | +4.0 | [−8.0, +16.0] | 1.000 |
| **ALL** | **100** | **72.0** | **64.0** | **−8.0** | [−16.0, 0.0] | 0.096 |

## Generated tokens

| domain | n | direct | sandbox | ratio | 95% CI |
|---|---:|---:|---:|---:|---:|
| biomedicine | 25 | 169 | 466 | **2.75×** | [1.31, 7.95] |
| chemistry | 25 | 1465 | 1267 | 0.86× | [0.59, 1.32] |
| math | 25 | 1406 | 887 | 0.63× | [0.29, 1.40] |
| physics | 25 | 1320 | 1059 | 0.80× | [0.45, 1.36] |
| **ALL** | **100** | **1090** | **920** | **0.84×** | [0.62, 1.16] |

## What actually happened

The overall −8.0 pp is not significant (p=0.096). **The math result is**
(−32 pp, p=0.008), and it is the only domain that moves. Directionally this
matches the paper's own Qwen3-4B row (−13.5 on maths), but the mechanism is not
"computing is worse than reasoning".

### The sandbox arm mostly did not use the sandbox

| domain | episodes with **0** bash commands | mean commands | acc when used | acc when unused |
|---|---:|---:|---:|---:|
| math | 8/25 (32%) | 1.4 | 0.65 | **0.38** |
| physics | 9/25 (36%) | 2.0 | 0.69 | 0.78 |
| chemistry | 10/25 (40%) | 1.8 | 0.53 | 0.80 |
| biomedicine | 19/25 (76%) | 0.8 | 0.50 | 0.68 |
| **ALL** | **46/100 (46%)** | 1.5 | 0.61 | 0.67 |

Mean file edits across all 100 episodes: **0.0**. The model never used
`file_editor` for anything but reading.

On maths — and only on maths — actually running code helped: 0.65 when it ran a
command, 0.38 when it did not. On every other domain the relationship inverts,
and on biomedicine the model correctly declined to use tools 76% of the time,
because "which hormone regulates X" is recall, not computation.

### It reasoned less, and did not reliably substitute computation

| domain | direct tokens | sandbox tokens |
|---|---:|---:|
| math | 1406 | **887** (−37%) |
| physics | 1320 | 1059 (−20%) |

The sandbox arm produced substantially *less* reasoning on maths, and in a third
of episodes replaced it with nothing at all. Those episodes score 0.38. That is
the −32 pp.

Representative failures (`direct` right, `sandbox` wrong):

- `math/8215` — 3 turns, **0 commands**, stalled, forced answer wrong
- `math/8836` — 1 turn, **0 commands**, answered in prose, wrong
- `math/8356` — 1 command, and the command was
  `echo "The value of the digit 5 in 24,513 is in the hundreds place..."` —
  the shell used to *narrate*, not to compute

### The efficiency claim inverts on recall tasks

Overall 0.84× sits at the top of the paper's reported 0.49–0.84× band, but the
per-domain split matters more: **biomedicine is 2.75×**, i.e. nearly three times
*more* expensive. The baseline answers those with a bare letter (169 tokens);
the sandbox arm pays for tool-calling ceremony (466 tokens) to reach the same
place. A sandbox is only cheap where there is something to compute.

## The confound, stated plainly

**This result should not be read as "sandboxes hurt small models."** The sandbox
system prompt in this repo says:

> You are expected to *compute* answers rather than derive them in your head

That wording actively discourages the chain-of-thought carrying the baseline's
88% on maths, while a 4B model only converts it into actual computation about
two-thirds of the time. The measured effect is therefore at least partly a
property of **that prompt**, not of the environment.

The cleaner claim this run supports: *a sandbox prompt that trades reasoning for
tool use is a bad trade at 4B, because the model does not reliably complete the
trade.*

The obvious next experiment — cheap, ~30 min — is a third arm with a sandbox
prompt that offers the tools without discouraging reasoning. If maths recovers,
the effect is prompt-driven; if it does not, it is environmental. Until that is
run, the −32 pp should be attributed to the combination, not the sandbox.

## Caveats

- n=100 paired. A simulated genuine +8 pp effect at this size reads as +13.3 pp
  with CI [−3.3, +30.0], p=0.185. Only the maths effect is large enough to
  resolve here; treat every other row as noise.
- Chemistry had **zero discordant pairs** (both arms right or wrong on the same
  25 items). With temperature 0 and a model that ignored the tools in 40% of
  those episodes, near-identical behaviour is plausible — but a CI of exactly
  [0, 0] is a coincidence worth remembering rather than a finding.
- 3/100 sandbox episodes ended without calling `finish`; 31 answered in prose
  and were recovered by the extractor.
- Single seed, single model, one benchmark.
