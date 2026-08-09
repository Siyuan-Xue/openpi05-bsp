import contextlib
import io
from pathlib import Path
import unittest

from scripts import compare_libero_phase1


class CompareLiberoPhaseOneCliTest(unittest.TestCase):
    def test_parser_requires_exactly_ten_runs_and_three_explicit_artifact_paths(self):
        parser = compare_libero_phase1.build_parser()
        arguments = [
            "run-a",
            "run-b",
            "run-c",
            "run-d",
            "run-e",
            "run-f",
            "run-g",
            "run-h",
            "run-i",
            "run-j",
            "--bsp-verification",
            "verify.json",
            "--norm-comparison",
            "norm.json",
            "--output-dir",
            "report",
        ]

        parsed = parser.parse_args(arguments)

        self.assertEqual(parsed.run_dirs, [Path(value) for value in arguments[:10]])
        self.assertEqual(parsed.bsp_verification, Path("verify.json"))
        self.assertEqual(parsed.norm_comparison, Path("norm.json"))
        self.assertEqual(parsed.output_dir, Path("report"))

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(arguments[1:])

        help_text = parser.format_help().lower()
        self.assertIn("ten", help_text)
        for milestone in ("0k", "1k", "2k", "5k", "10k"):
            self.assertIn(milestone, help_text)
        self.assertNotIn("20k", help_text)
        self.assertNotIn("30k", help_text)


if __name__ == "__main__":
    unittest.main()
