import importlib
import subprocess
import sys
import unittest


MODULES_WITH_HELP = (
    "spectralix_benchmark.evaluation.student_validation",
    "spectralix_benchmark.evaluation.llm_judge",
    "spectralix_benchmark.evaluation.run_full_matrix",
)

MODULES_IMPORTABLE = (
    "spectralix_benchmark",
    "spectralix_benchmark.evaluation.cli.student_validation",
    "spectralix_benchmark.evaluation.cli.llm_judge",
    "spectralix_benchmark.evaluation.cli.run_full_matrix",
    "spectralix_benchmark.evaluation.pipeline.student_validation",
    "spectralix_benchmark.evaluation.pipeline.llm_judge",
    "spectralix_benchmark.evaluation.pipeline.run_full_matrix",
    "spectralix_benchmark.evaluation.io.jsonl",
)

SCRIPT_HELP_COMMANDS = (
    ("spectralix-student",),
    ("spectralix-judge",),
    ("spectralix-matrix",),
    ("spectralix-materialize",),
    ("spectralix-build",),
    ("spectralix-build-levels",),
    ("spectralix-build-paper-eval",),
)


class EvaluationCliHelpTests(unittest.TestCase):
    def test_legacy_module_entrypoints_support_help(self):
        for module_name in MODULES_WITH_HELP:
            with self.subTest(module_name=module_name):
                result = subprocess.run(
                    [sys.executable, "-m", module_name, "--help"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_console_scripts_support_help(self):
        for command in SCRIPT_HELP_COMMANDS:
            with self.subTest(command=command):
                result = subprocess.run(
                    ["uv", "run", *command, "--help"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("usage:", result.stdout.lower())


class EvaluationImportabilityTests(unittest.TestCase):
    def test_evaluation_split_modules_import(self):
        for module_name in MODULES_IMPORTABLE:
            with self.subTest(module_name=module_name):
                imported = importlib.import_module(module_name)
                self.assertIsNotNone(imported)


if __name__ == "__main__":
    unittest.main()
