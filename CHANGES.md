# Changes on this fork

This is a fork of [`killkli/minimax-h3-colab-skill`](https://github.com/killkli/minimax-h3-colab-skill) — a well-designed Codex skill that turns local reference images into MiniMax H3 Ref2VA clips through Google Colab. The design is good; the first version had bugs that cost real compute units before they announced themselves. This branch fixes all 17 issues filed as #1–#17 on this fork.

- Forked from upstream `7768ebf`
- Branch: `fix/colab-runner-hardening`
- Diff: 12 files, +907 / −189
- Nothing here is merged upstream yet, and the upstream project carries no licence grant, so this fork adds no `LICENSE` either. Relicensing is the upstream author's call.

## The three that mattered

**1. Cold-start timeout discarded the weight download (#1, P0).** The first `colab exec` inherited a 3600 s budget. A fresh runtime has to clone ComfyUI, install dependencies and pull 39–59 GiB of weights before it renders a single frame, so job one predictably died at the 60-minute mark and took the download with it. `run_batch` now gives the first job its own `--setup-timeout` (default 10800 s, or `COLAB_SETUP_TIMEOUT`) and later jobs `--timeout`, and derives `H3_JOB_TIMEOUT_SECONDS` so the notebook stops itself a margin *before* the CLI does. A slow job now reports where it stalled instead of returning a bare timeout.

**2. The notebook deleted finished renders (#3, P1).** The `ffprobe` audio-stream assertion ran *before* the MP4 was copied to the CLI output path, so a video-only render was destroyed by the code meant to deliver it — and the runner asserted audio too, turning a usable clip into a failed job. Publish first, verify second: the copy and the Colab download now precede the media check, missing audio is a warning, and `--require-audio` restores the strict behaviour when you want it.

**3. `--session` killed the runtime it was told to reuse (#2, P0).** `COLAB_SESSION_NAME` was documented as reusable but implemented as "always `colab new`, always stop in cleanup" — so a live session was clobbered or stopped mid-queue while its work was still running. Sessions now track ownership: a session the runner created is named, created and stopped; a session you named is reused and left running, with `--stop-on-complete` as the explicit opt-out.

## Everything else

| # | Sev | Symptom | Fix |
| --- | --- | --- | --- |
| 4 | P1 | One transient HTTP error in the render-poll loop aborted a long job | Tolerates `MAX_POLL_FAILURES = 20` consecutive errors with backoff |
| 5 | P1 | `ref_images` keys exempted from the graph schema check, so a renamed slot silently dropped reference images | Exemption narrowed to the keys that genuinely vary |
| 6 | P1 | Log filter *dropped* every line over ~320 chars — tracebacks included | `redact()` truncates at `MAX_LOG_CHARS = 400` with a `…[+N chars]` marker and keeps diagnostics in the progress file |
| 7 | P1 | Full prompt echoed to `exec` stdout, contradicting `SKILL.md` | Prompt digest by default, full text only with `H3_VERBOSE_PROMPT=1`; prompt bodies suppressed in logs |
| 8 | P2 | No seed control from the shell launcher, so single runs were unreproducible | `--seed` / `H3_SEED` threaded to the notebook |
| 9 | P2 | `single` hardcoded `progress_path=None`, so a crash left no state | `--progress` for `single`, defaulted to `<stem>.progress.json` beside the output |
| 10 | P2 | UTF-8 BOM uploaded inside the model prompt | `read_prompt_text()` uses `utf-8-sig`; a mid-string BOM is rejected |
| 11 | P2 | Duration/seed validation errors didn't name the job | Errors attributed by id and index, before any session is requested |
| 12 | P2 | A malformed explicit job id was randomised, bypassing duplicate detection | Explicit ids are validated and rejected when malformed |
| 13 | P2 | `check_cli()` was dead code despite documented version validation | `colab version` checked up front; warning outside `VALIDATED_CLI_MINOR = {6, 7}`; version recorded in progress |
| 14 | P2 | `--gpu` unvalidated, so `--gpu --help` reached the CLI as an option | `validate_gpu()` as `type=` on every `--gpu`: `[A-Za-z0-9][A-Za-z0-9_-]{0,31}` |
| 15 | P2 | Job prompts written in plaintext beside the manifest | Private `mkdtemp(prefix="h3_work_")` work root, `chmod 0o700` before anything is written |
| 16 | P2 | No CI, no licence provenance | `.github/workflows/ci.yml` + provenance stated here and in the READMEs; **licence decision still open** |
| 17 | P2 | Test suite was POSIX-locked; 4 of 8 tests failed on Windows with no skip guard | `tests/fake_colab_cli.py` single-source fake with a per-platform launcher; only the genuine bash test stays POSIX-gated |

## Behaviour and interface changes

| Knob | Where | Effect |
| --- | --- | --- |
| `--setup-timeout` / `COLAB_SETUP_TIMEOUT` | runner, launcher | Budget for job one (default 10800 s) |
| `--require-audio` | runner, launcher | Fail instead of warn on a silent MP4 |
| `--seed` / `H3_SEED` | runner, launcher | Pin the sampler for reproducible clips |
| `--progress` | runner (`single` too) | JSON state beside the output by default |
| `--stop-on-complete` | runner | Stop a session you handed in with `--session` |
| `H3_JOB_TIMEOUT_SECONDS` | notebook | Inner deadline, set by the runner |
| `H3_VERBOSE_PROMPT` | notebook | Print the whole prompt into the log |

## Verify without a GPU

The suite needs no Colab session, no credentials and no compute units — `colab` and `ffprobe` are replaced by local fakes:

```bash
python3 -m unittest discover -s tests -v    # 39 tests, OK (skipped=4) on Windows
python3 scripts/runner.py --help
./run_colab_inference.sh --help
```

The three `test_colab_cli_contract.py` checks and the bash launcher test skip off Linux, and run against a real `google-colab-cli` install in CI.

## Known limits

- The Colab CLI is Linux/macOS-only, so the runner is too; on Windows the suite runs, a real batch does not.
- Sufficient compute units do not guarantee a GPU or high-memory runtime is allocated; allocation failures are reported plainly, not retried.
- #16 stays open on purpose: no `LICENSE` was added to a fork of unlicensed third-party code.
