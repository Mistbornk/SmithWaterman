#!/usr/bin/env python3
"""Exercise benchmark CLI/CSV and FASTA tooling using isolated temporary files."""
import csv
import io
import pathlib
import subprocess
import sys
import tempfile
import unittest

BENCHMARK = sys.argv.pop(1)
ROOT = pathlib.Path(__file__).resolve().parent


class FlowTests(unittest.TestCase):
    def test_benchmark_csv_and_reproducibility(self):
        for backend in ("scalar", "simd"):
            rows = []
            for _ in range(2):
                result = subprocess.run([
                    BENCHMARK, "--backend", backend, "--batch", "5", "--length", "64",
                    "--threads", "2", "--repetitions", "3", "--warmup", "1",
                ], check=True, capture_output=True, text=True)
                parsed = list(csv.DictReader(io.StringIO(result.stdout)))
                self.assertEqual(len(parsed), 1)
                row = parsed[0]
                self.assertEqual(row["backend"], backend)
                self.assertEqual(row["threads_effective"], "2")
                self.assertLessEqual(float(row["min_us"]), float(row["median_us"]))
                self.assertLessEqual(float(row["median_us"]), float(row["p95_us"]))
                self.assertGreater(float(row["pairs_per_s"]), 0)
                rows.append(row)
            self.assertEqual(rows[0]["checksum"], rows[1]["checksum"])

    def test_invalid_cli(self):
        for args in (["--batch", "0"], ["--length", "-1"], ["--seed", "4294967296"],
                     ["--threads", "0"], ["--threads", "257"], ["--identity", "101"],
                     ["--mode", "unknown"], ["--mode", "single", "--threads", "2"],
                     ["--backend", "unknown"], ["--length", "1000000000"],
                     ["--repetitions", "0"], ["--batch"]):
            with self.subTest(args=args):
                result = subprocess.run([BENCHMARK, *args], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"benchmark:", result.stderr)

    def test_fasta_reproducibility_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory(prefix="sw-flow-") as tmp:
            paths = [pathlib.Path(tmp) / "one", pathlib.Path(tmp) / "two"]
            for path in paths:
                subprocess.run([sys.executable, str(ROOT / "generate_fasta.py"),
                                "100", "70", "--seed", "29", "--out-dir", str(path)],
                               check=True, capture_output=True)
            for name in ("ref.fasta", "alt.fasta"):
                self.assertEqual((paths[0] / name).read_bytes(), (paths[1] / name).read_bytes())
            ref = (paths[0] / "ref.fasta").read_text().splitlines()[1]
            query = (paths[0] / "alt.fasta").read_text().splitlines()[1]
            self.assertEqual(sum(a != b for a, b in zip(ref, query)), 30)
            original = (paths[0] / "ref.fasta").read_bytes()
            result = subprocess.run([sys.executable, str(ROOT / "generate_fasta.py"),
                                     "100", "70", "--out-dir", str(paths[0])], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((paths[0] / "ref.fasta").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
