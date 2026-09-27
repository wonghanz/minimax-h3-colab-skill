from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runner


# The fake `colab` CLI is a Python script exposed through a per-platform launcher,
# so these tests run wherever Python runs. Only run_colab_inference.sh stays
# POSIX-only: google-colab-cli itself supports Linux and macOS.
POSIX_ONLY = os.name == "posix"

FAKE_COLAB_CLI = Path(__file__).with_name("fake_colab_cli.py")

FAKE_FFPROBE = (
    "import json, os\n"
    "streams = [s for s in os.environ.get('FAKE_FFPROBE_STREAMS', 'video,audio').split(',') if s]\n"
    "print(json.dumps({'streams': [{'codec_type': stream} for stream in streams]}))\n"
)


def install_fake_executable(bin_dir: Path, name: str, script: Path) -> None:
    """Expose a Python script on PATH under the executable name `name`."""
    if os.name == "posix":
        launcher = bin_dir / name
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
    else:
        launcher = bin_dir / f"{name}.bat"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        ffprobe_script = self.fake_bin / "ffprobe_cli.py"
        ffprobe_script.write_text(FAKE_FFPROBE, encoding="utf-8")
        install_fake_executable(self.fake_bin, "colab", FAKE_COLAB_CLI)
        install_fake_executable(self.fake_bin, "ffprobe", ffprobe_script)
        self.state_dir = self.root / "state"
        self.env_patch = patch.dict(os.environ, {
            "PATH": os.pathsep.join([str(self.fake_bin), os.environ.get("PATH", "")]),
            "FAKE_COLAB_STATE_DIR": str(self.state_dir),
        })
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.temp.cleanup()

    def manifest(self, job_count: int = 2, prompt_file: bool = False, directory: Path | None = None) -> Path:
        base = directory or self.root
        base.mkdir(parents=True, exist_ok=True)
        jobs = []
        for index in range(job_count):
            image = base / f"reference-{index}.png"
            image.write_bytes(b"fake image")
            job = {
                "id": f"job-{index}",
                "title": f"clip {index}",
                "reference_images": [str(image)],
                "prompt": f"complete prompt <Picture 1> for clip {index}",
                "duration_seconds": 8,
                "seed": 100 + index,
                "output_name": f"clip-{index}",
            }
            if prompt_file:
                prompt = base / f"prompt-{index}.txt"
                prompt.write_text(job.pop("prompt"), encoding="utf-8")
                job["prompt_file"] = str(prompt)
            jobs.append(job)
        manifest = base / "manifest.json"
        manifest.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
        return manifest

    def calls(self) -> list[dict[str, object]]:
        path = self.state_dir / "calls.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def run_batch(self, manifest: Path, **kwargs) -> dict[str, object]:
        options = {
            "session": None, "gpu": "A100", "high_mem": True, "stop_on_complete": False,
            "progress_path": self.root / "progress.json",
            "output_dir": self.root / "outputs", "exec_timeout": 100,
        }
        options.update(kwargs)
        return runner.run_batch(manifest, **options)

    # ------------------------------------------------------------------ parsing

    def test_usage_parser_handles_labelled_colab_cli_output(self) -> None:
        result = runner.parse_usage("Current balance: 1,234.50 compute units\nUsage rate: 12.00/hr\nActive assignments: 3\n")
        self.assertEqual(result["balance"], 1234.5)
        self.assertEqual(result["active_assignments"], 3)

    def test_usage_command_reads_colab_through_cli(self) -> None:
        result = runner.get_usage()
        self.assertEqual(result["balance"], 100.0)
        self.assertEqual(result["rate_per_hour"], 0.0)
        self.assertEqual(result["active_assignments"], 0)

    def test_guided_prompt_builds_shots_and_checks_picture_mapping(self) -> None:
        prompt = runner.compose_ref2va_prompt(
            subject_definitions="<Subject 1> is the presenter in <Picture 1>.",
            summary="A short introduction.",
            retention_analysis="<Subject 1> remains consistent across [Shot 1].",
            shots=[
                {"start_seconds": 0, "description": "Faces the camera.", "dialogue": "大家好！"},
                {"start_seconds": 4.2, "description": "Gestures toward a terminal card."},
            ],
            overall_soundscape="Quiet studio ambience.",
            non_diegetic_music="N/A",
            image_count=1,
            duration_seconds=8,
            detailed_description="Opening: a clean blue studio with a terminal card beside the presenter.",
        )
        self.assertIn("<d>[Chinese] 大家好！</d>", prompt)
        self.assertIn("[Shot 2] At 00:04.200", prompt)
        self.assertIn("Opening: a clean blue studio", prompt)
        with self.assertRaisesRegex(ValueError, "Picture"):
            runner.compose_ref2va_prompt(
                subject_definitions="<Subject 1> uses <Picture 2>.",
                summary="Summary", retention_analysis="Retention",
                shots=[{"start_seconds": 0, "description": "Faces camera."}],
                overall_soundscape="Room tone", non_diegetic_music="N/A",
                image_count=1, duration_seconds=8,
            )

    # ------------------------------------------------------- issue #17 / #13 CLI

    def test_cli_version_guard_warns_outside_validated_range(self) -> None:
        self.assertIsNone(runner.cli_version_warning("colab 0.7.4"))
        self.assertIsNone(runner.cli_version_warning("colab 0.6.1"))
        self.assertIn("outside the validated range", runner.cli_version_warning("colab 0.9.0"))
        self.assertIn("outside the validated range", runner.cli_version_warning("colab 1.0.0"))
        self.assertIn("Could not parse", runner.cli_version_warning("colab nightly"))

    def test_batch_records_the_cli_version_it_ran_against(self) -> None:
        result = self.run_batch(self.manifest(1))
        self.assertEqual(result["colab_cli_version"], "colab 0.7.4")
        with patch.dict(os.environ, {"FAKE_COLAB_VERSION": "colab 0.9.0"}):
            warned = self.run_batch(self.manifest(1))
        self.assertTrue(any("outside the validated range" in item for item in warned["warnings"]))

    # ----------------------------------------------------- issue #1 timeout budget

    def test_first_exec_gets_the_setup_budget_and_later_execs_the_job_budget(self) -> None:
        result = self.run_batch(self.manifest(2), exec_timeout=100, setup_timeout=10800)
        self.assertEqual(result["status"], "completed")
        execs = [call for call in self.calls() if call["command"] == "exec"]
        self.assertEqual([float(call["cli_timeout"]) for call in execs], [10800.0, 100.0])
        for call in execs:
            self.assertLess(float(call["job_timeout"]), float(call["cli_timeout"]),
                            "the notebook deadline must fire before the CLI kills the exec")

    def test_inner_deadline_never_drops_below_the_minimum(self) -> None:
        self.run_batch(self.manifest(1), exec_timeout=30, setup_timeout=45)
        execs = [call for call in self.calls() if call["command"] == "exec"]
        self.assertEqual(float(execs[0]["job_timeout"]), 60.0)

    # ---------------------------------------------------- issue #2 session owner

    def start_live_session(self, name: str) -> None:
        (self.state_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (self.state_dir / "sessions" / name).write_text("live", encoding="utf-8")

    def test_live_named_session_is_reused_and_left_running(self) -> None:
        self.start_live_session("shared")
        result = self.run_batch(self.manifest(1), session="shared")
        calls = self.calls()
        self.assertEqual(sum(call["command"] == "new" for call in calls), 0)
        self.assertEqual(sum(call["command"] == "stop" for call in calls), 0)
        self.assertEqual(result["session_ownership"], "reused")
        self.assertEqual(result["session_status"], "active")

    def test_absent_named_session_is_created_and_stopped(self) -> None:
        result = self.run_batch(self.manifest(1), session="fresh")
        calls = self.calls()
        self.assertEqual(sum(call["command"] == "new" for call in calls), 1)
        self.assertEqual(sum(call["command"] == "stop" for call in calls), 1)
        self.assertEqual(result["session_ownership"], "created")
        self.assertEqual(result["session_status"], "stopped")

    def test_reused_session_is_stopped_only_when_asked(self) -> None:
        self.start_live_session("shared")
        self.run_batch(self.manifest(1), session="shared", stop_on_complete=True)
        calls = self.calls()
        self.assertEqual(sum(call["command"] == "new" for call in calls), 0)
        self.assertEqual(sum(call["command"] == "stop" for call in calls), 1)

    # ------------------------------------------------------------------- batches

    def test_batch_reuses_one_session_and_preserves_prompt_file_content(self) -> None:
        manifest = self.manifest(2, prompt_file=True)
        result = self.run_batch(manifest)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["completed_count"], 2)
        self.assertEqual(result["session_status"], "stopped")
        self.assertTrue(all(Path(item["output"]).is_file() for item in result["results"]))
        calls = self.calls()
        self.assertEqual(sum(call["command"] == "new" for call in calls), 1)
        self.assertEqual(sum(call["command"] == "exec" for call in calls), 2)
        self.assertEqual(sum(call["command"] == "stop" for call in calls), 1)
        uploads = [call for call in calls if call["command"] == "upload"]
        self.assertEqual(len({call["remote"] for call in uploads}), 4)
        exec_call = next(call for call in calls if call["command"] == "exec")
        uploaded = self.state_dir / "remote" / str(exec_call["prompt"]).lstrip("/")
        self.assertEqual(uploaded.read_text(encoding="utf-8"), "complete prompt <Picture 1> for clip 0")

    def test_failed_later_job_keeps_completed_video_and_stops_session(self) -> None:
        with patch.dict(os.environ, {"FAKE_FAIL_PREFIX": "MiniMax_H3_job-1"}):
            result = self.run_batch(self.manifest(2), high_mem=False)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["completed_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertTrue(Path(result["results"][0]["output"]).is_file())
        self.assertEqual(sum(call["command"] == "stop" for call in self.calls()), 1)

    @unittest.skipUnless(POSIX_ONLY, "run_colab_inference.sh is a POSIX shell entry point")
    def test_shell_launcher_accepts_multiple_local_images_and_prompt_file(self) -> None:
        image_one = self.root / "ref-one.png"
        image_two = self.root / "ref-two.jpg"
        prompt = self.root / "prompt.txt"
        output = self.root / "from-shell.mp4"
        image_one.write_bytes(b"fake png bytes")
        image_two.write_bytes(b"fake jpeg bytes")
        prompt_content = "complete prompt with <Picture 1> and <Picture 2>\n"
        prompt.write_text(prompt_content, encoding="utf-8")
        project = Path(__file__).resolve().parents[1]
        subprocess.run([
            "bash", str(project / "run_colab_inference.sh"),
            "--image", str(image_one), "--image", str(image_two),
            "--prompt", str(prompt), "--output", str(output), "--seed", "4242",
        ], env=os.environ.copy(), capture_output=True, text=True, check=True, timeout=180)
        self.assertEqual(output.read_bytes(), b"fake video bytes")
        calls = self.calls()
        self.assertEqual(sum(call["command"] == "new" for call in calls), 1)
        self.assertEqual(sum(call["command"] == "exec" for call in calls), 1)
        self.assertEqual(sum(call["command"] == "stop" for call in calls), 1)
        exec_call = next(call for call in calls if call["command"] == "exec")
        self.assertEqual(len(exec_call["refs"]), 2)
        # issue #8: the launcher has to be able to pin a seed
        self.assertEqual(exec_call["seed"], "4242")
        uploaded_prompt = self.state_dir / "remote" / str(exec_call["prompt"]).lstrip("/")
        self.assertEqual(uploaded_prompt.read_text(encoding="utf-8"), prompt_content)
        # issue #9: single must leave a progress record behind
        self.assertTrue((output.parent / "from-shell.progress.json").is_file())

    # ----------------------------------------------------------- issue #3 audio

    def write_fake_mp4(self) -> Path:
        media = self.root / "clip.mp4"
        media.write_bytes(b"fake video bytes")
        return media

    def test_video_only_render_warns_instead_of_failing(self) -> None:
        media = self.write_fake_mp4()
        with patch.dict(os.environ, {"FAKE_FFPROBE_STREAMS": "video"}):
            warnings = runner.verify_mp4(media)
            self.assertEqual(len(warnings), 1)
            self.assertIn("no audio stream", warnings[0])
            with self.assertRaisesRegex(ValueError, "no audio stream"):
                runner.verify_mp4(media, require_audio=True)

    def test_render_without_video_stream_still_fails(self) -> None:
        media = self.write_fake_mp4()
        with patch.dict(os.environ, {"FAKE_FFPROBE_STREAMS": "audio"}):
            with self.assertRaisesRegex(ValueError, "no video stream"):
                runner.verify_mp4(media)

    def test_missing_output_is_still_an_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            runner.verify_mp4(self.root / "absent.mp4")

    def test_batch_without_audio_keeps_the_render_and_reports_a_warning(self) -> None:
        with patch.dict(os.environ, {"FAKE_FFPROBE_STREAMS": "video"}):
            result = self.run_batch(self.manifest(1))
        self.assertEqual(result["status"], "completed")
        self.assertTrue(Path(result["results"][0]["output"]).is_file())
        self.assertTrue(any("no audio stream" in item for item in result["warnings"]))

    def test_batch_can_treat_missing_audio_as_a_failure_without_losing_the_file(self) -> None:
        with patch.dict(os.environ, {"FAKE_FFPROBE_STREAMS": "video"}):
            result = self.run_batch(self.manifest(1), require_audio=True)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["jobs"][0]["status"], "failed")
        self.assertTrue(Path(result["jobs"][0]["output"]).is_file())

    # ------------------------------------------------ issues #6 / #7 log hygiene

    def test_redact_truncates_long_lines_instead_of_dropping_them(self) -> None:
        long_error = "RuntimeError: node validation failed " + "x" * 900
        result = runner.redact(long_error)
        self.assertTrue(result.startswith("RuntimeError: node validation failed"))
        self.assertIn("[+537 chars]", result)
        self.assertLess(len(result), 460)

    def test_redact_suppresses_prompt_body(self) -> None:
        self.assertIn("prompt content suppressed", runner.redact("the presenter says <d>[Chinese] 你好</d>"))
        self.assertIn("prompt content suppressed", runner.redact("subject_definitions: <Subject 1> in <Picture 1>"))

    def test_command_error_output_is_redacted(self) -> None:
        error = runner.ColabCommandError("generate clip", 7, ["plain diagnostic", "x" * 900])
        self.assertIn("plain diagnostic", str(error))
        self.assertIn("[+500 chars]", str(error))

    def test_prompt_text_never_reaches_the_progress_file(self) -> None:
        secret = "SECRET-PROMPT-TEXT"
        manifest = self.manifest(1)
        jobs = json.loads(manifest.read_text(encoding="utf-8"))["jobs"]
        jobs[0]["prompt"] = f"{secret} <Picture 1> " + "detail " * 200
        manifest.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
        result = self.run_batch(manifest)
        self.assertEqual(result["status"], "completed")
        # The fake notebook echoes the prompt, so the redaction has to hold.
        derived = (self.root / "progress.json").read_text(encoding="utf-8")
        self.assertNotIn(secret, derived)
        self.assertIn("prompt content suppressed", derived)

    def test_failure_diagnostics_reach_the_progress_file(self) -> None:
        with patch.dict(os.environ, {"FAKE_FAIL_PREFIX": "MiniMax_H3_job-0"}):
            result = self.run_batch(self.manifest(1))
        self.assertEqual(result["failed_count"], 1)
        self.assertIn("simulated inference failure", str(result["jobs"][0]["error"]))

    # ------------------------------------------------------- issue #10 BOM

    def test_prompt_file_bom_is_stripped_but_body_preserved(self) -> None:
        path = self.root / "bom.txt"
        path.write_bytes("complete prompt <Picture 1>".encode("utf-8-sig"))
        self.assertEqual(runner.read_prompt_text(path), "complete prompt <Picture 1>")
        plain = self.root / "plain.txt"
        plain.write_bytes("complete prompt <Picture 1>".encode("utf-8"))
        self.assertEqual(runner.read_prompt_text(plain), "complete prompt <Picture 1>")

    def test_mid_string_bom_is_rejected(self) -> None:
        path = self.root / "mid.txt"
        path.write_bytes("prompt \ufeffwith stray mark".encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "byte-order mark"):
            runner.read_prompt_text(path)

    # ------------------------------------------- issues #11 / #12 manifest errors

    def test_duration_and_seed_errors_name_the_job(self) -> None:
        data = json.loads(self.manifest(2).read_text(encoding="utf-8"))
        data["jobs"][1]["duration_seconds"] = "eight"
        with self.assertRaisesRegex(ValueError, "Job 2 duration"):
            runner.resolve_jobs(data, self.root / "output")
        data["jobs"][1]["duration_seconds"] = 8
        data["jobs"][1]["seed"] = "abc"
        with self.assertRaisesRegex(ValueError, "Job 2 seed"):
            runner.resolve_jobs(data, self.root / "output")

    def test_malformed_explicit_job_id_is_rejected_not_randomised(self) -> None:
        data = json.loads(self.manifest(1).read_text(encoding="utf-8"))
        data["jobs"][0]["id"] = "intro v2"
        with self.assertRaisesRegex(ValueError, "Job 1 id"):
            runner.resolve_jobs(data, self.root / "output")
        generated = runner.safe_job_id(None, 3)
        self.assertEqual(len(generated), 16)
        self.assertRegex(generated, r"^[A-Za-z0-9_-]{16}$")

    def test_resolve_jobs_rejects_duplicate_job_ids_and_outputs(self) -> None:
        data = json.loads(self.manifest(2).read_text(encoding="utf-8"))
        data["jobs"][1]["id"] = data["jobs"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicates job id"):
            runner.resolve_jobs(data, self.root / "output")
        data["jobs"][1]["id"] = "job-1"
        data["jobs"][1]["output_path"] = str(self.root / "same.mp4")
        data["jobs"][0]["output_path"] = str(self.root / "same.mp4")
        with self.assertRaisesRegex(ValueError, "same output path"):
            runner.resolve_jobs(data, self.root / "output")

    def test_resolve_jobs_rejects_picture_tags_outside_uploaded_references(self) -> None:
        data = json.loads(self.manifest(1).read_text(encoding="utf-8"))
        data["jobs"][0]["prompt"] += " <Picture 2>"
        with self.assertRaisesRegex(ValueError, "Picture"):
            runner.resolve_jobs(data, self.root / "output")

    # ---------------------------------------------------------- issue #14 gpu

    def test_gpu_names_are_validated_before_any_session_request(self) -> None:
        self.assertEqual(runner.validate_gpu("A100"), "A100")
        self.assertEqual(runner.validate_gpu("nvidia_h100"), "nvidia_h100")
        for bad in ("--high-mem", "A 100", "", "a" * 33):
            with self.assertRaises(ValueError):
                runner.validate_gpu(bad)

    def test_bad_gpu_never_reaches_colab_new(self) -> None:
        with self.assertRaises(ValueError):
            self.run_batch(self.manifest(1), gpu="--high-mem")
        self.assertFalse((self.state_dir / "calls.jsonl").exists())

    # -------------------------------------------------------- issue #15 temp dir

    def test_job_prompts_are_not_written_next_to_the_manifest(self) -> None:
        job_dir = self.root / "project"
        manifest = self.manifest(1, prompt_file=True, directory=job_dir)
        output_dir = job_dir / "clips"
        def files_in(directory: Path) -> list[str]:
            return sorted(entry.name for entry in directory.iterdir() if entry.is_file())

        before = files_in(job_dir)
        self.run_batch(manifest, output_dir=output_dir)
        self.assertEqual(files_in(job_dir), before)
        self.assertEqual(files_in(output_dir), ["clip-0_job-0.mp4"])


if __name__ == "__main__":
    unittest.main()
