from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "assets" / "MiniMax_H3_Turbo_Colab.ipynb"


def code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


class NotebookTests(unittest.TestCase):
    """The notebook runs on a paid GPU session, so break there cost compute."""

    def test_every_code_cell_compiles(self) -> None:
        for number, source in enumerate(code_cells(), start=1):
            with self.subTest(cell=number):
                ast.parse(source)

    def test_cli_deadline_comes_from_the_runner(self) -> None:
        # Issue #1: the notebook must stop on its own deadline, before the CLI
        # kills the exec, so the failure is reported instead of a bare timeout.
        source = "\n".join(code_cells())
        self.assertIn("H3_JOB_TIMEOUT_SECONDS", source)

    def test_render_is_published_before_the_media_check(self) -> None:
        # Issue #3: an already paid-for render must reach the download path even
        # when the stream inspection below it fails.
        publishers = [source for source in code_cells() if "COLAB_CLI_OUTPUT_PATH" in source]
        self.assertEqual(len(publishers), 1)
        source = publishers[0]
        self.assertLess(
            source.index("shutil.copy2(video_path, CLI_OUTPUT_PATH)"),
            source.index("ffprobe"),
            "the MP4 has to be copied out before it is inspected",
        )
        self.assertLess(
            source.index("shutil.copy2(video_path, CLI_OUTPUT_PATH)"),
            source.index("files.download(str(video_path))"),
            "the web branch has to keep the render before downloading it",
        )


if __name__ == "__main__":
    unittest.main()
