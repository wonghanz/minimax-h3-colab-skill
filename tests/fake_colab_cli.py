#!/usr/bin/env python3
"""Stand-in for the ``colab`` command line, used only by the tests.

``google-colab-cli`` ships for Linux and macOS, so the suite drives this fake
through two transports: as an executable on PATH (POSIX, which exercises the
runner's real subprocess plumbing) and in process (any OS, which exercises the
runner's decisions). Both transports share this one dispatch implementation.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

USAGE_OUTPUT = (
    "Current balance: 100.00 compute units\n"
    "Usage rate: 0.00/hr\n"
    "Active assignments: 0"
)


def _flag(args: list[str], name: str, default: str | None = None) -> str | None:
    return args[args.index(name) + 1] if name in args else default


def _record(root: Path, kind: str, **values: object) -> None:
    with (root / "calls.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": kind, **values}) + "\n")


def main(argv: list[str]) -> int:
    args = [arg for arg in argv if not arg.startswith("--auth=")]
    if not args:
        print("fake colab: no command", file=sys.stderr)
        return 2
    command, rest = args[0], args[1:]
    root = Path(os.environ["FAKE_COLAB_STATE_DIR"])
    sessions = root / "sessions"
    remote = root / "remote"
    sessions.mkdir(parents=True, exist_ok=True)
    remote.mkdir(parents=True, exist_ok=True)

    if command == "usage":
        print(USAGE_OUTPUT)
    elif command == "version":
        print(os.environ.get("FAKE_COLAB_VERSION", "colab 0.7.4"))
    elif command == "status":
        name = _flag(rest, "--session")
        if (sessions / str(name)).is_file():
            print(f"{name} | https://colab.example/{name} | GPU A100 | IDLE")
        else:
            print(f"[colab] Session '{name}' not found.")
    elif command == "new":
        name = str(_flag(rest, "--session"))
        if os.environ.get("FAKE_NEW_FAILS") == name:
            print("[colab] Allocation refused (precondition failed).", file=sys.stderr)
            return 1
        (sessions / name).write_text("live", encoding="utf-8")
        _record(root, "new", args=rest)
        print("Session created")
    elif command == "stop":
        name = str(_flag(rest, "--session"))
        (sessions / name).unlink(missing_ok=True)
        _record(root, "stop", args=rest)
        print("Session stopped")
    elif command == "upload":
        source, target = Path(rest[-2]), remote / rest[-1].lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        _record(root, "upload", remote=rest[-1], source=source.name)
    elif command == "exec":
        envs = dict(item.split("=", 1) for item in (
            rest[index + 1] for index, value in enumerate(rest[:-1]) if value == "--env"
        ))
        output_path = envs["H3_OUTPUT_PATH"]
        prompt_text = (remote / envs["H3_PROMPT_FILE"].lstrip("/")).read_text(encoding="utf-8")
        _record(
            root, "exec", output=output_path,
            refs=json.loads(envs["H3_REFERENCE_IMAGES"]),
            prompt=envs["H3_PROMPT_FILE"], seed=envs.get("H3_SEED"),
            cli_timeout=_flag(rest, "--timeout"),
            job_timeout=envs.get("H3_JOB_TIMEOUT_SECONDS"),
        )
        # A real notebook echoes its prompt and long tracebacks; the runner has
        # to keep both out of the progress file.
        print("prompt 已準備：" + prompt_text.replace("\n", " "))
        if envs.get("H3_OUTPUT_PREFIX") == os.environ.get("FAKE_FAIL_PREFIX"):
            print("simulated inference failure " + "x" * 900, file=sys.stderr)
            return 7
        target = remote / output_path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake video bytes")
    elif command == "download":
        source, target = remote / rest[-2].lstrip("/"), Path(rest[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        _record(root, "download", remote=rest[-2], target=target.name)
    else:
        print(f"unexpected fake command: {command}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
