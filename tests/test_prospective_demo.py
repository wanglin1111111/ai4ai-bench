"""tests/test_prospective_demo.py — prospective_demo gate 管线正反向自测（零依赖，unittest）。

运行：
  python -m unittest discover -s tests -v
  python tests/test_prospective_demo.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import prospective_demo as demo  # noqa: E402

DEMO = ROOT / "src" / "prospective_demo.py"


def make_raw(benchmark="webshop", code="def hook(ctx, nb):\n    return {'effect': 'pass'}"):
    patch = {"benchmark": benchmark, "description": "d",
             "actions": [{"type": "add_code_hook", "hook": "on_before_action", "code": code}]}
    return "<patch>" + json.dumps(patch) + "</patch>"


class TestRunGate(unittest.TestCase):
    def test_offline_sample_valid(self):
        entry = demo.run_gate(demo.OFFLINE_RAW, "test")
        self.assertTrue(entry["format_validation"]["valid"])
        self.assertEqual(entry["decision"], "PENDING_OUTCOME_EVAL")
        self.assertIn("complexity", entry)

    def test_reject_without_patch_block(self):
        entry = demo.run_gate("no patch block at all", "test")
        self.assertFalse(entry["format_validation"]["valid"])
        self.assertEqual(entry["decision"], "REJECT_INVALID")

    def test_reject_benchmark_mismatch(self):
        entry = demo.run_gate(make_raw(benchmark="alfworld"), "test")
        self.assertEqual(entry["decision"], "REJECT_INVALID")
        self.assertEqual(entry["format_validation"]["error"], "benchmark mismatch")

    def test_reject_forbidden_syntax(self):
        entry = demo.run_gate(make_raw(code="import os\ndef hook(ctx, nb):\n    return 1"), "test")
        self.assertEqual(entry["decision"], "REJECT_INVALID")
        self.assertIn("forbidden", entry["format_validation"]["error"])

    def test_reject_empty_actions(self):
        raw = "<patch>" + json.dumps({"benchmark": "webshop", "actions": []}) + "</patch>"
        entry = demo.run_gate(raw, "test")
        self.assertEqual(entry["decision"], "REJECT_INVALID")


class TestCli(unittest.TestCase):
    def test_offline_exit_zero_and_log(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "log.json"
            r = subprocess.run([sys.executable, str(DEMO), "--offline", "--log", str(log)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            data = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual(len(data["prospective_entries"]), 1)
            self.assertTrue(data["prospective_entries"][0]["format_validation"]["valid"])

    def test_missing_results_readable_error(self):
        env = dict(os.environ)
        env.pop("HARNESS_R1_RESULTS", None)
        r = subprocess.run([sys.executable, str(DEMO)], capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("未找到上游 results.json", r.stdout + r.stderr)

    def test_reject_exit_one(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "bad.txt"
            raw.write_text("no patch block", encoding="utf-8")
            log = Path(td) / "log.json"
            r = subprocess.run([sys.executable, str(DEMO), "--raw-file", str(raw), "--log", str(log)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()