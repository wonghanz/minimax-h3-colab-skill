---
name: minimax-h3-colab
description: Create short MiniMax H3 reference-to-video clips from one or more local images through Google Colab CLI. Use when a user asks for image-guided H3 video generation or a sequential batch of clips.
---

# MiniMax H3 on Colab

Use the bundled notebook and `scripts/runner.py` to generate Ref2VA video from local reference images. The Colab CLI must be installed and authenticated with an available compute-unit balance and GPU allocation.

## Workflow

1. Inspect the requested images and any supplied prompt file. Keep image order stable: image 1 maps to `<Picture 1>`, image 2 to `<Picture 2>`, through image 9.
2. If the user supplied a prompt file, pass it through unchanged. Otherwise write a complete Ref2VA prompt with `subject_definitions`, `summary`, `retention_analysis`, `detailed_description` shot blocks, `overall_soundscape`, and `non_diegetic_music`; include dialogue in its shot with `<d>[Chinese] ...</d>` when requested. Use the bundled runner's `compose_ref2va_prompt` helper for structured shot input.
3. Check Colab access and balance with `python3 scripts/runner.py usage --json`. Sufficient compute units do not guarantee that Colab will allocate the requested GPU or high-memory runtime. Report allocation failures plainly.
4. Put the jobs in a UTF-8 JSON manifest and run one batch. The runner uploads each job's images and prompt, executes the bundled notebook sequentially on the same live session, downloads each MP4, and stops the session when it created it. If reusing a session, pass `--stop-on-complete` when the user wants it stopped after the batch. The first job gets the setup budget because it installs ComfyUI and downloads the weights; later jobs get the per-job budget.
5. Confirm each output exists and report its local path, plus the seed and any warnings recorded in the progress file. Preserve completed clips if a later job fails.

Example manifest:

```json
{
  "jobs": [
    {
      "title": "intro",
      "reference_images": ["/absolute/path/girl.jpg"],
      "prompt_file": "/absolute/path/prompt.txt",
      "duration_seconds": 12,
      "output_name": "intro"
    }
  ]
}
```

Run from this skill directory:

```bash
python3 scripts/runner.py batch --manifest /absolute/path/jobs.json --gpu A100 --setup-timeout 10800 --timeout 3600 --output-dir /absolute/path/outputs --progress /absolute/path/outputs/progress.json
```

Use `--no-high-mem` when high-memory allocation is unavailable or not desired. For a named session, `--session SESSION` reuses it while `colab status` reports it live and leaves it running; otherwise the runner creates that name and stops it afterwards. Add `--stop-on-complete` if the requested workflow should end a reused session.

## Operational limits

- Each job requires 1–9 non-empty reference images, a non-empty UTF-8 prompt, and a duration from 4–15 seconds. Job ids must match `[A-Za-z0-9_-]{1,48}`.
- A job may set `seed`; otherwise the runner draws one and records it in the progress file, so report the seed with the clip.
- Keep all jobs in one batch to reuse the same Colab session and loaded ComfyUI/model state.
- A downloaded MP4 without an audio stream is kept and reported as a warning; pass `--require-audio` only when silent output must fail.
- Progress files and error messages carry truncated diagnostics and never the prompt body. Do not paste prompt text into logs or status output.
- Do not blindly retry a timed-out `colab exec`: the remote kernel may still be working. The runner stops the session during cleanup and records completed jobs before reporting the failure.
- Do not expose OAuth credentials or runtime tokens in prompts, manifests, browser state, or logs.
- The bundled `assets/MiniMax_H3_Turbo_Colab.ipynb` is the inference notebook used by the runner. The model checkpoints it downloads carry their own licences; this skill grants no rights to them.
