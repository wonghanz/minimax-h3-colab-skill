# MiniMax H3 Colab Skill

This repository is a complete, standalone Codex skill for creating short MiniMax H3 Ref2VA videos from local reference images through Google Colab. Clone it, install the skill, authenticate the Colab CLI, and invoke the included runner or shell launcher. The repository contains:

- `SKILL.md`: the instructions Codex loads when this skill is selected;
- `scripts/runner.py`: the session, usage, upload, batch, download, and cleanup runner;
- `assets/MiniMax_H3_Turbo_Colab.ipynb`: the notebook executed on the remote Colab runtime;
- `run_colab_inference.sh`: a convenient single-video launcher;
- `install.sh`: a portable, non-destructive skill installer;
- `tests/`: offline tests using a fake Colab CLI, plus notebook and CLI-contract checks.

The local machine only prepares and uploads inputs. Model inference runs on the Colab GPU and the finished MP4 is downloaded back to the path you choose.

## Requirements

- A working Codex installation. The skill is installed into `$CODEX_HOME/skills` when `CODEX_HOME` is set, otherwise into `~/.codex/skills`.
- Python 3.11 or newer for the bundled runner. The runner uses only the Python standard library.
- [`uv`](https://docs.astral.sh/uv/) for installing the Colab CLI, or another supported way to put `colab` on `PATH`.
- [`google-colab-cli`](https://pypi.org/project/google-colab-cli/). The runner was validated with Colab CLI 0.7.4 and uses the documented `version`, `usage`, `new`, `status`, `upload`, `exec`, `download`, and `stop` commands; it reports a warning when the installed CLI is outside the validated 0.6–0.7 series. The current CLI release requires Python 3.12 or newer; `uv` can install that interpreter separately from the runner's Python 3.11+ requirement. The CLI supports Linux and macOS, so the runner does too.
- A Google account with access to Colab compute units and a GPU shape that can be allocated. An A100 or equivalent high-memory runtime may require the appropriate Colab plan and available balance.

Downloads are verified with `ffprobe` when it is installed: a missing or unreadable MP4 fails the job, and a video-only MP4 is kept with a warning, because it is already a paid-for result. Pass `--require-audio` to treat a silent MP4 as a failure instead. Without `ffprobe`, the runner only checks that the file exists and is non-empty.

## Clone and install the skill

```bash
git clone <repository-url> minimax-h3-colab-skill
cd minimax-h3-colab-skill
./install.sh
```

The installer copies the required skill files to:

```text
$CODEX_HOME/skills/minimax-h3-colab
```

When `CODEX_HOME` is unset, the destination is `~/.codex/skills/minimax-h3-colab`.

Use an explicit skills directory when testing or when your Codex configuration is elsewhere:

```bash
./install.sh --dest /absolute/path/to/codex/skills
```

The default operation is idempotent and non-destructive. If the destination already exists, the installer leaves it unchanged and exits successfully. To replace it deliberately, use `--force`; the old directory is first moved to a timestamped `.backup.*` path so it can be recovered:

```bash
./install.sh --force
```

The installer uses a temporary directory and an atomic rename for a new installation. It never copies the repository's Git metadata, tests, README files, or generated output into the installed skill; only `SKILL.md`, `scripts/`, and `assets/` are installed.

After installation, start a new Codex turn or reload the skill list if your Codex client caches available skills. The skill name is `minimax-h3-colab`.

## Install and authenticate Colab CLI

Install the CLI as a user tool:

```bash
uv python install 3.12
uv tool install --python 3.12 google-colab-cli
```

Confirm that it is available:

```bash
colab version
```

The runner defaults to OAuth2. Trigger the first authorization and inspect the account balance with:

```bash
colab --auth=oauth2 usage
```

Follow the URL and code instructions printed by the CLI. The token is stored by the CLI in its normal local configuration; do not put credentials, tokens, or browser state in this repository, a prompt, or a job manifest. If your environment already uses Google Application Default Credentials, choose the alternative provider explicitly:

```bash
COLAB_AUTH=adc colab --auth=adc usage
```

The runner passes `--auth="$COLAB_AUTH"` to every Colab command. Its default is `oauth2`.

## Single-video inference

Create a UTF-8 text file such as `prompt.txt`, then run:

```bash
./run_colab_inference.sh \
  --image /absolute/path/reference_1.png \
  --image /absolute/path/reference_2.jpg \
  --prompt /absolute/path/prompt.txt \
  --output /absolute/path/intro.mp4
```

The first image is exposed to the notebook as `<Picture 1>`, the second as `<Picture 2>`, and so on. Pass 1–9 non-empty image files. The prompt file must be non-empty UTF-8 text and is uploaded unchanged. If `--output` is omitted, the MP4 is written next to the first reference image with a `_minimax_h3.mp4` suffix.

The default clip length is 12 seconds. Set `H3_DURATION_SECONDS` to a value from 4 through 15:

```bash
H3_DURATION_SECONDS=8 ./run_colab_inference.sh \
  --image /absolute/path/reference.png \
  --prompt /absolute/path/prompt.txt
```

The shell launcher requests an A100 high-memory runtime by default. These environment variables change the request:

| Variable | Default | Meaning |
| --- | --- | --- |
| `COLAB_AUTH` | `oauth2` | Colab CLI auth strategy: `oauth2` or `adc` |
| `COLAB_GPU` | `A100` | GPU name passed to `colab new` |
| `COLAB_HIGH_MEM` | `1` | Set to `0` to omit `--high-mem` |
| `COLAB_EXEC_TIMEOUT` | `3600` | Per-notebook execution timeout once the runtime is warm |
| `COLAB_SETUP_TIMEOUT` | `10800` | Execution timeout for the first notebook run, which installs ComfyUI and downloads the weights |
| `COLAB_SESSION_NAME` | generated | Reusable session name for the runner |
| `H3_DURATION_SECONDS` | `12` | Clip duration, from 4 to 15 seconds |
| `H3_SEED` | random | Seed to reuse; `--seed` on the launcher overrides it |

The launcher also accepts `--seed` for reproducible clips and `--progress` for a state file. Without `--progress`, progress is written next to the output as `<output>.progress.json`. Prompts, log lines longer than 400 characters, and remote output paths are recorded there in redacted form.

## Batch inference

For several clips, keep all jobs in one manifest so they reuse one Colab session and the loaded model. Example `jobs.json`:

```json
{
  "jobs": [
    {
      "id": "intro",
      "title": "Presenter introduction",
      "reference_images": [
        "/absolute/path/girl-front.png",
        "/absolute/path/girl-side.png"
      ],
      "prompt_file": "/absolute/path/intro-prompt.txt",
      "duration_seconds": 8,
      "output_name": "intro"
    },
    {
      "id": "demo",
      "title": "CLI demo",
      "reference_images": ["/absolute/path/girl-front.png"],
      "prompt": "A concise Ref2VA prompt referring to <Picture 1>.",
      "duration_seconds": 12,
      "output_name": "demo"
    }
  ]
}
```

Run the batch from the cloned repository:

```bash
python3 scripts/runner.py batch \
  --manifest /absolute/path/jobs.json \
  --gpu A100 \
  --setup-timeout 10800 \
  --timeout 3600 \
  --output-dir /absolute/path/outputs \
  --progress /absolute/path/outputs/progress.json
```

The first job on a fresh runtime pays for the ComfyUI install and the 39–59 GiB weight download, so it gets `--setup-timeout` (or `COLAB_SETUP_TIMEOUT`); later jobs on the same session get `--timeout`. The notebook stops on its own deadline a few minutes before the Colab CLI kills the execution, so a slow job reports where it stalled instead of surfacing a bare timeout.

The runner validates all local inputs before starting, and it checks the Colab CLI version first. It uploads each job's references and prompt, executes the bundled notebook, downloads and verifies the MP4, and continues with the next job. A later failure does not delete completed outputs, and prompts are staged in a private temporary directory rather than next to the manifest.

Session handling depends on who created the session: with no `--session`, the runner names, creates, and always stops its own session; with `--session NAME`, it reuses that session when `colab status` reports it live and leaves it running, or creates it and stops it afterwards. Add `--stop-on-complete` to stop a reused session once the queue finishes.

The installed skill can also be invoked directly without the cloned repository:

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/minimax-h3-colab/scripts/runner.py" \
  batch --manifest /absolute/path/jobs.json --output-dir /absolute/path/outputs
```

## Prompt and image rules

- Each job needs 1–9 non-empty local reference images.
- Image order is stable and is the only mapping used for `<Picture N>` tags.
- A prompt file is read as UTF-8 (a leading byte-order mark is stripped, and a stray one inside the text is rejected) and sent as the complete prompt; the runner does not translate, summarize, or rewrite it.
- A prompt cannot refer to a picture number greater than the number of images uploaded for that job.
- Each video duration must be between 4 and 15 seconds.
- A job may pin `seed` (0 to 2^64-1); omit it to draw a fresh one, which is reported in the progress file so a result can be replayed.
- Job `id` values must match `[A-Za-z0-9_-]{1,48}`; an invalid explicit id is rejected rather than silently replaced, because it becomes the remote file name.
- For guided prompts in another application, use the runner's `compose_ref2va_prompt` helper to produce `subject_definitions`, `summary`, `retention_analysis`, ordered shot blocks, `overall_soundscape`, and `non_diegetic_music`.
- The reference image guide is available in the [MiniMax H3 Ref2VA prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).

Reference images describe visual identity and appearance. They do not automatically create shot timestamps; describe timing and camera changes in the prompt.

## Direct notebook use

`assets/MiniMax_H3_Turbo_Colab.ipynb` is the exact notebook the runner uploads and executes. It can also be opened manually in Colab for inspection or debugging. The runner's environment variables select reference mode, remote image paths, prompt path, duration, seed, output path, and the notebook-side deadline (`H3_JOB_TIMEOUT_SECONDS`). Set `H3_VERBOSE_PROMPT=1` in a notebook cell to print the full prompt instead of its length and SHA-256 digest. Keep the notebook copy in this repository with the runner so the two stay compatible.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `colab` is missing | Run `uv tool install google-colab-cli`, then ensure the uv tool bin directory is on `PATH`. The CLI only runs on Linux and macOS. |
| OAuth or usage fails | Run `colab --auth=oauth2 usage` interactively and complete the printed Google authorization flow. |
| GPU allocation fails | Check Colab plan, compute-unit balance, requested GPU, and high-memory availability. Try `--no-high-mem` or another supported GPU. |
| The first job times out | The runtime was probably still installing ComfyUI or downloading weights. Raise `--setup-timeout` (or `COLAB_SETUP_TIMEOUT`), or reuse one session across jobs. |
| A batch times out | Do not blindly retry while the remote kernel may still be running. The runner attempts to stop a session it owns and records completed jobs in progress state. |
| Prompt picture validation fails | Match `<Picture N>` to the 1-based order of the 1–9 `reference_images` entries. |
| Progress log shows suppressed text | Long lines are truncated to 400 characters and prompt bodies are replaced by a placeholder on purpose; the full prompt is uploaded verbatim. |
| Output exists but is silent | The render is kept and reported as a warning. Install `ffprobe` to inspect the streams, and pass `--require-audio` to fail such a job instead. |

## Offline validation

No GPU or Colab session is needed to run the repository tests:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/runner.py --help
./run_colab_inference.sh --help
```

The tests replace `colab` and `ffprobe` with local fakes, so they do not spend compute units or access credentials. The notebook tests compile every code cell and check that the render is copied out before it is inspected. A few tests are skipped where they cannot apply: the shell launcher needs a POSIX shell, and the Colab CLI contract tests need `colab` on `PATH`. GitHub Actions runs the same suite on Linux with the CLI installed.

## Scope and safety

This repository does not contain Google credentials, tokens, model weights, or generated videos. Colab sessions consume the account's compute units. Do not place secrets in prompts, manifests, logs, or uploaded files. Review the requested GPU and timeout before starting a real batch.

The notebooks download checkpoints from `Comfy-Org/MiniMax-H3`, `drbaph/MiniMax-H3-Turbo-Lora-ComfyUI`, and the Qwen3-VL text encoder, and run them through ComfyUI and ComfyUI-VideoHelperSuite. Each of those artifacts carries its own licence and acceptable-use terms, and this repository grants no rights to them; check them before any use beyond personal experiments. This repository itself publishes no licence statement for its scripts and documentation.
