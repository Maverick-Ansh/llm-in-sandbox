# llm-in-sandbox

A from-scratch reproduction and stress-test of **"Computer Environments Elicit
General Agentic Intelligence in LLMs"** ([arXiv:2601.16206][paper]) on commodity
hardware — two Tesla T4s.

The paper's claim is unusually clean: give a model nothing but a **minimal
computer** — a bash shell, a file editor, and a way to say "done" — and general
reasoning ability improves across maths, physics, chemistry and biomedicine, at
a *fraction* of the tokens a long chain of thought would burn. No retraining, no
elaborate tool suite.

This repo builds that environment properly, runs the comparison, and pushes on
the parts the paper leaves soft.

[paper]: https://arxiv.org/abs/2601.16206

---

## What's here

```
src/sandbox_lab/
  sandbox/     the computer: persistent shell, file editor, isolation, seccomp
  agent/       the loop: model <-> sandbox, two run modes
  evals/       benchmarks, deterministic grading, resumable sweeps
scripts/       serve.py, run_sweep.py, report.py
notebooks/     colab_driver.ipynb (reproduce) + session_log_2xT4.ipynb (record)
tests/         including enforced-ablation checks with positive controls
```

### The three tools

Faithful to the paper: `bash`, `file_editor` (view/create/str_replace/insert),
and `finish`. Nothing else. The minimality is the hypothesis, so adding a
`python` tool or a web-search tool would be testing a different claim.

### The shell is a PTY, not a pipe

This is the part most implementations get subtly wrong, and it is worth
explaining because the failure is silent.

The paper specifies a **10-second soft timeout**. "Soft" is load-bearing: when a
command overruns, we must kill *that command* and leave the session alive with
its working directory, exported variables and shell functions intact. A pipe
based implementation typically kills the shell, and the agent's state silently
resets mid-episode — which reads later as "the model got confused about paths".

Doing it correctly needs the kernel's terminal line discipline:

- `set -m` puts each command in its own process group and installs it as the
  PTY's **foreground** group.
- Writing `\x03` to the PTY master makes the tty driver deliver `SIGINT` to that
  group only.
- bash is the session leader in a *different* group, so it survives — exactly
  what happens when a human presses Ctrl-C.

Commands are also **sourced from a file** rather than piped in as text. Piping
breaks the moment a command contains a heredoc: the sentinel line appended to
detect completion gets swallowed as heredoc content, and the read loop hangs
until the timeout. `test_command_with_heredoc_survives_framing` pins this.

### Isolation is built from namespaces, not Docker

Colab has no Docker daemon, so `UnshareBackend` builds the isolation from the
same kernel primitives Docker uses — PID, mount, UTS, IPC and optionally network
namespaces — with no daemon and no image. `DockerBackend` is used when a daemon
*is* available.

This also upgrades the paper's ablations. The paper removes a meta-capability by
**describing its absence in the prompt**. That cannot distinguish "the model
obeyed" from "the model ignored us and it worked anyway". Here each capability
is removed for real:

| capability | paper | here |
|---|---|---|
| external resource access | prompt says no network | namespace or seccomp: `pip install` genuinely fails |
| file management | prompt discourages files | read-only bind mount: `open(...,"w")` genuinely fails |
| code execution | tool withheld | tool withheld |

Every "the capability is gone" test is paired with a **positive control** — the
same probe must succeed with the capability enabled. A test that only checks the
network is down also passes on a host with no network at all, and would certify
an ablation that never happened.

### Getting enforcement on a host that forbids namespaces

Colab denies `unshare(2)` even to uid 0 — the container drops `CAP_SYS_ADMIN`,
so every namespace flag returns `Operation not permitted`. That would leave the
paper's most distinctive capability unenforceable on the only hardware this
project runs on.

`SeccompBackend` is the way through. Since Linux 3.5 an **unprivileged** process
may install a seccomp-BPF syscall filter provided it first sets
`PR_SET_NO_NEW_PRIVS` — the promise that it cannot gain privileges through a
later `execve`, which is what makes the operation safe to allow without
capabilities. The filter denies `socket(AF_INET/AF_INET6)` with `EACCES` and is
inherited across `fork` and `execve`, so it covers every process the agent
spawns, including a subshell. `AF_UNIX` is deliberately untouched: the ablation
removes *external resource* access, not local IPC.

Measured on Colab: `pip install` fails, `socket.create_connection` raises
`PermissionError`, unix sockets and file writes are unaffected.

It is honestly weaker than a namespace — it filters syscalls, it does not
virtualise the machine, so there is no PID or mount isolation. It therefore
**refuses** `file_management=False` rather than pretending: a read-only path
needs a mount namespace, and accepting it silently would produce a run labelled
as an enforced ablation in which the model could still write files.

Backend order is `docker > unshare > seccomp > local`, and the *same* backend is
used for every condition in a comparison — otherwise the measured delta would be
confounded by the isolation mechanism changing underneath it. `select_backend`
refuses to fall back to the unisolated backend when a capability is ablated; a
silent downgrade would produce a results table that looks fine and means
nothing.

---

## Hardware reality

The target is 2× Tesla T4 (15 GB each, 30 GB total), compute capability **7.5**.
That rules out a lot, and it shapes every choice below:

- **no bf16** — everything runs fp16
- **no FP8, no Marlin** — 4-bit AWQ is the only useful quantisation
- **no FlashAttention-2** (needs sm80+) — vLLM falls back to FlexAttention
- **no NVLink** — the two cards are `PHB` (PCIe host bridge), so tensor-parallel
  across them is slow; one model per card beats one model across two

Serving is vLLM 0.11.0 with prefix caching on, which matters more here than
usual: an agent loop re-sends the whole transcript every turn, so turn *N*
shares a long prefix with turn *N−1* and only the new tail needs prefilling.

---

## Deviations from the paper

Stated plainly, because they bound what these results can claim:

| | paper | here | why |
|---|---|---|---|
| max turns | 100 | 20–30 | T4 throughput |
| max gen/turn | 65,536 | 2,048 | 24k context ceiling on a 15 GB card |
| context | — | 24,576 | KV cache for 32k did not fit alongside fp16 weights |
| models | GPT-5, Claude-Sonnet-4.5, Qwen3-Coder-30B-A3B, Qwen3-4B | Qwen3-4B-Instruct-2507 (+ ladder) | what fits |
| science benchmark | GPQA | MMLU-Pro | GPQA is gated; MMLU-Pro is one dataset spanning all four domains, so cross-domain comparisons are not confounded by differing question style |

**Why start at 4B.** The paper's headline is +15.5 on Qwen3-Coder-30B. Its most
interesting row is Qwen3-4B: **−13.5 on maths**. The sandbox *hurts* small
models. That negative result is the one claim at this scale we can actually
interrogate — and "where is the capability threshold at which a computer
environment starts paying off" is the question the paper raises but does not
answer.

---

## Methodology notes

Things that would quietly invalidate the comparison, and what is done about
them:

- **One grader, both modes.** Deterministic — normalise, exact match, then
  numeric and symbolic equivalence. No LLM judge: a judge rewards tidiness, and
  the sandbox mode produces tidier answers (`42`) than the prose baseline (`the
  answer is 42 units`), so it would credit the sandbox for formatting.
- **The tolerance is asymmetric on purpose.** A computed `0.3333333333` must
  match a reference of `1/3`, or we penalise the exact behaviour under study.
  But two *integers* must compare exactly, or `1000000` and `1000001` differ by
  1e-6 relative and wrong AIME answers grade correct.
- **The baseline is not handicapped.** `direct` mode gets 4× the per-turn token
  budget in one shot, because capping it at the agent's per-turn budget would
  manufacture the efficiency result for free.
- **Paired comparison.** The summary compares only tasks *both* modes attempted;
  otherwise the delta is confounded with which subset each mode finished.
- **Harness artefacts are recovered, not scored.** Qwen3-4B reliably writes
  `finish(849)` as prose instead of emitting a tool call. Unrecovered, that
  registers as a stall — a formatting slip masquerading as a capability
  difference, and one that can only ever penalise the sandbox mode since the
  baseline emits no tool calls at all.

---

## Four measurement bugs, and which way each one lied

The first real run surfaced four bugs. None crashed anything. Every one of them
would have shown up purely as a wrong number in the results table, and three of
the four pushed in a direction that would have *confirmed* the paper's finding
for small models. They are listed here because "we ran it and got a delta" is
worth very little without knowing what was checked.

| bug | who it hurt | size |
|---|---|---|
| Extractor required a `FINAL ANSWER:` label | **baseline only** | 17/99 episodes; accuracy 0.596 → 0.700 |
| Tool arguments read only from `arguments` | **sandbox only** | every `bash` call failed → episode burned all 20 turns |
| Forced-answer call not counted as a turn | **sandbox only** | its tokens vanished, understating sandbox usage |
| 240s request deadline | **baseline only** | 7/99 episodes scored wrong for generating too much |

**The label bug is the instructive one.** MMLU-Pro's own prompt says "answer
with the letter of the correct option only", and the model obeyed it —
returning `C`, two tokens, which the extractor read as *no answer at all*. The
sandbox arm answers through the `finish` tool and never touches that path, so
the bug was structurally incapable of affecting it. Left in, it would have
handed the sandbox a ~10 point advantage made entirely of formatting.

**The tool-argument bug is its mirror image.** Qwen3-4B emits
`{"name": "bash", "command": ...}` with parameters at the top level rather than
nested under `"arguments"`. Reading only `arguments` yielded `{}`, so every
command failed with "command is required" and the episode exhausted its turn
budget. In a results table that is indistinguishable from a small model that
cannot use tools — which is precisely the paper's reported finding at 4B. It
would have looked like a successful reproduction.

The general lesson: in a two-arm comparison, any bug in a code path that only
one arm exercises is an effect-size bug, not a crash. Both arms therefore share
the grader and the extractor, and the loop-level tests assert that
`generated_tokens` means the same thing on both sides.

---

## Running it

```bash
pip install -e '.[dev]'
pytest tests/ -q                      # editor + grader tests run anywhere;
                                      # shell/sandbox tests need Linux

# serve (T4: fp16, 24k context)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 --served-model-name qwen3-4b \
  --dtype float16 --max-model-len 24576 --gpu-memory-utilization 0.93 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes

# sweep (resumable: rerun the same command to continue)
python scripts/run_sweep.py --benchmark mmlu_pro --n 40 \
    --model qwen3-4b --out runs/qwen3-4b-mmlu
```

`notebooks/` contains the Colab driver used to produce the results below.

---

## Notebooks

- **`notebooks/colab_driver.ipynb`** — clean, reproducible driver: install,
  serve, sweep, report. Start here.
- **`notebooks/session_log_2xT4.ipynb`** — the actual exploratory session,
  kept because the dead ends are the useful part. Records each probe and what it
  forced: the `-9` shell bug being ours rather than the platform's, `unshare`
  being denied while seccomp is permitted, and FlexAttention costing ~5x
  throughput.

---

## Results

Qwen3-4B-Instruct-2507, MMLU-Pro, **100 paired items**. Full write-up:
[`results/qwen3-4b-mmlu-pro.md`](results/qwen3-4b-mmlu-pro.md).

| domain | direct | sandbox | delta (pp) | 95% CI | p |
|---|---:|---:|---:|---:|---:|
| math | 88.0 | 56.0 | **−32.0** | [−52, −16] | **0.008** |
| physics | 68.0 | 72.0 | +4.0 | [−8, +16] | 1.000 |
| chemistry | 64.0 | 64.0 | 0.0 | [0, 0] | 1.000 |
| biomedicine | 68.0 | 64.0 | −4.0 | [−24, +16] | 1.000 |
| **ALL** | **72.0** | **64.0** | **−8.0** | [−16, 0] | 0.096 |

Tokens overall **0.84×** — but that average hides the interesting part:
maths 0.63×, **biomedicine 2.75×**. A sandbox is only cheap where there is
something to compute; on recall questions the baseline answers with one letter
(169 tokens) and the sandbox pays 466 for the tool-calling ceremony.

**Directionally this matches the paper's own Qwen3-4B row (−13.5 on maths), but
the mechanism is not "computing is worse than reasoning".** In 46% of episodes
the model ran **no commands at all**, and mean file edits were 0.0. On maths it
scored 0.65 when it did run a command and **0.38 when it did not** — while
producing 37% less reasoning than the baseline (887 vs 1406 tokens). It gave up
the chain-of-thought and only sometimes bought computation with it.

> **Read the confound before quoting the number.** This repo's sandbox prompt
> says *"you are expected to compute answers rather than derive them in your
> head"*, which actively discourages the reasoning carrying the baseline's 88%.
> The honest claim is not "sandboxes hurt small models" but "a prompt that
> trades reasoning for tool use is a bad trade at 4B, because the model does not
> reliably complete the trade." A third arm with a non-discouraging prompt would
> separate the two; it has not been run.

Only the maths effect is large enough to resolve at n=100 — a simulated genuine
+8pp effect reads as +13.3pp with CI [−3.3, +30.0], p=0.185, so treat every
other row as noise.

Also established, measured rather than assumed:

| finding | evidence |
|---|---|
| Enforced network ablation works where namespaces are denied | 7/7 seccomp tests on Colab, each with a positive control |
| `unshare` is unavailable on Colab | `Operation not permitted` as uid 0 (no `CAP_SYS_ADMIN`) |
| Attention backend dominates T4 throughput | FlexAttention ~22 tok/s → TRITON_ATTN 51–69 tok/s per card; ~120 tok/s across both |

---

## Licence

Apache-2.0.
