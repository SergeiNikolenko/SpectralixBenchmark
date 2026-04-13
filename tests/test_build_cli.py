import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spectralix_benchmark.build import level_benchmark_files
from spectralix_benchmark.build import paper_eval_subsets


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_module(module_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module_name, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


class BuildModulePathResolutionTests(unittest.TestCase):
    def test_level_builder_path_resolution(self) -> None:
        self.assertEqual(level_benchmark_files.REPO_ROOT, REPO_ROOT)
        self.assertEqual(level_benchmark_files.BENCHMARK_DIR, REPO_ROOT / "benchmark")
        self.assertEqual(level_benchmark_files.EXTERNAL_SOURCES_DIR, REPO_ROOT / "external_sources")

    def test_paper_builder_path_resolution(self) -> None:
        self.assertEqual(paper_eval_subsets.REPO_ROOT, REPO_ROOT)
        self.assertEqual(paper_eval_subsets.BENCHMARK_DIR, REPO_ROOT / "benchmark")
        self.assertEqual(paper_eval_subsets.LEVEL_A_POOL, REPO_ROOT / "benchmark" / "level_a.jsonl")
        self.assertEqual(paper_eval_subsets.LEVEL_B_POOL, REPO_ROOT / "benchmark" / "level_b.jsonl")
        self.assertEqual(paper_eval_subsets.LEVEL_C_POOL, REPO_ROOT / "benchmark" / "level_c.jsonl")


class BuildCliTests(unittest.TestCase):
    def test_level_builder_help(self) -> None:
        result = _run_module("spectralix_benchmark.build.level_benchmark_files", "--help")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--benchmark-dir", result.stdout)
        self.assertIn("--external-sources-dir", result.stdout)
        self.assertIn("--dry-run", result.stdout)

    def test_paper_builder_help(self) -> None:
        result = _run_module("spectralix_benchmark.build.paper_eval_subsets", "--help")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--benchmark-dir", result.stdout)
        self.assertIn("--level-a-pool", result.stdout)
        self.assertIn("--dry-run", result.stdout)

    def test_build_cli_help(self) -> None:
        result = _run_module("spectralix_benchmark.build.cli", "--help")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("levels", result.stdout)
        self.assertIn("paper-eval", result.stdout)

    def test_level_builder_fails_fast_when_inputs_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            benchmark_dir = base / "benchmark"
            external_sources_dir = base / "external_sources"
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            external_sources_dir.mkdir(parents=True, exist_ok=True)

            result = _run_module(
                "spectralix_benchmark.build.level_benchmark_files",
                "--benchmark-dir",
                str(benchmark_dir),
                "--external-sources-dir",
                str(external_sources_dir),
                "--dry-run",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing required input paths", result.stderr)

    def test_paper_builder_fails_fast_when_pools_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            benchmark_dir = base / "benchmark"
            benchmark_dir.mkdir(parents=True, exist_ok=True)

            result = _run_module(
                "spectralix_benchmark.build.paper_eval_subsets",
                "--benchmark-dir",
                str(benchmark_dir),
                "--dry-run",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing required input pools", result.stderr)

