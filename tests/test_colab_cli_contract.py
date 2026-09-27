"""The runner shells out to google-colab-cli, so pin the flags it depends on.

Upstream renames a flag and every batch fails at the first `colab exec`, hours
into a paid session. These checks are skipped when the CLI is not installed so
the rest of the suite stays dependency-free.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest

COLAB = shutil.which("colab")
ANSI_SEQUENCE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Rich splits "--session" into styled runs, so a literal substring check on raw
# help output never matches. Ask for plain text and strip whatever remains.
PLAIN_ENV = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"}


def strip_markup(text: str) -> str:
    return ANSI_SEQUENCE.sub("", text).replace("\x1b", "")


class HelpParsingTests(unittest.TestCase):
    def test_ansi_styled_rich_help_still_exposes_flag_names(self) -> None:
        # Captured from CI: typer renders each option with per-run colour codes.
        styled = (
            "\x1b[1m\x1b[0m\x1b[1;36m-\x1b[0m\x1b[1;36m-session\x1b[0m   "
            "\x1b[1;32m-s\x1b[0m      Session name\n"
            "\x1b[1;36m-\x1b[0m\x1b[1;36m-high\x1b[0m\x1b[1;36m-mem\x1b[0m          Request a high-RAM machine\n"
        )
        for flag in ("--session", "--high-mem"):
            with self.subTest(flag=flag):
                self.assertIn(flag, strip_markup(styled))


@unittest.skipUnless(COLAB, "google-colab-cli is not installed")
class ColabCliContractTests(unittest.TestCase):
    def help_for(self, *arguments: str) -> str:
        result = subprocess.run(
            [COLAB, "--auth=oauth2", *arguments, "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=PLAIN_ENV,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        return strip_markup(result.stdout + result.stderr)

    def test_documented_commands_are_available(self) -> None:
        listing = self.help_for()
        for command in ("usage", "new", "status", "upload", "download", "exec", "stop"):
            with self.subTest(command=command):
                self.assertIn(command, listing)

    def test_exec_accepts_the_flags_the_runner_uses(self) -> None:
        help_text = self.help_for("exec")
        for flag in ("--session", "--timeout", "--env", "--file"):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)

    def test_new_accepts_the_session_shaping_flags(self) -> None:
        help_text = self.help_for("new")
        for flag in ("--session", "--gpu", "--high-mem"):
            with self.subTest(flag=flag):
                self.assertIn(flag, help_text)


if __name__ == "__main__":
    unittest.main()
