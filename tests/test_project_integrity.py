from __future__ import annotations

import json
import py_compile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectIntegrityTests(unittest.TestCase):
    def test_required_project_files_exist(self) -> None:
        required = [
            "README.md",
            "requirements.txt",
            "data/raw/loanapproval.csv",
            "src/data.py",
            "src/modeling.py",
            "src/calibration.py",
            "src/abstention.py",
            "src/plots.py",
            "src/pipeline.py",
            "app/app.py",
        ]

        for relative_path in required:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).exists(), f"Missing {relative_path}")

    def test_python_source_files_compile(self) -> None:
        source_files = list((ROOT / "src").glob("*.py")) + [ROOT / "app" / "app.py"]
        self.assertTrue(source_files, "No Python source files found")

        for path in source_files:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                py_compile.compile(str(path), doraise=True)

    def test_requirements_are_line_separated(self) -> None:
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        dependencies = [line.strip() for line in requirements if line.strip() and not line.startswith("#")]

        self.assertGreaterEqual(len(dependencies), 5)
        for dependency in dependencies:
            with self.subTest(dependency=dependency):
                self.assertNotIn(" ", dependency, "Each dependency should be on its own line")

    def test_existing_json_outputs_are_valid_when_present(self) -> None:
        json_paths = [
            ROOT / "outputs" / "metrics_overall.json",
            ROOT / "outputs" / "abstention_policy.json",
            ROOT / "outputs" / "data_quality.json",
        ]

        for path in json_paths:
            if not path.exists():
                continue
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertIsInstance(data, dict)
