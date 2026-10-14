"""Execute every chapter and verify selected physical/circuit invariants."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["chapters"]
CACHE = {}


def chapter(number):
    if number not in CACHE:
        record = CATALOG[number - 1]
        spec = importlib.util.spec_from_file_location(f"book_chapter_{number}", ROOT / record["path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with contextlib.redirect_stdout(io.StringIO()):
            CACHE[number] = module.main()
    return CACHE[number]


class ChapterExecution(unittest.TestCase):
    """Each lab starts from a fresh module and must finish without an exception."""


def make_test(number):
    def test(self):
        self.assertIsInstance(chapter(number), dict)
    return test


for record in CATALOG:
    setattr(ChapterExecution, f"test_chapter_{record['number']:02d}", make_test(record["number"]))


class ScientificInvariants(unittest.TestCase):
    def test_teleportation_fidelity(self):
        env = chapter(11)
        fidelity = env["state_fidelity"](env["rho_bob"], env["target"])
        self.assertAlmostEqual(float(fidelity), 1.0, places=11)

    def test_manual_qft_matches_library(self):
        env = chapter(16)
        self.assertTrue(env["manual"].equiv(env["library"]))

    def test_grover_probability(self):
        env = chapter(15)
        probability = env["circuit_success"](3, "101", 2)
        self.assertAlmostEqual(probability, 0.9453125, places=11)

    def test_simon_empty_equations_and_secret(self):
        env = chapter(14)
        empty_solution = env["nonzero_null_vector"]([], 3)
        self.assertIsNotNone(empty_solution)
        self.assertNotEqual(empty_solution, "000")
        self.assertEqual(env["nonzero_null_vector"]([[0, 1, 0], [1, 0, 1]], 3), "101")

    def test_amplitude_damping_is_trace_preserving(self):
        env = chapter(19)
        operators = [env["K0"], env["K1"]]
        completeness = sum(k.conj().T @ k for k in operators)
        np.testing.assert_allclose(completeness, np.eye(2), atol=1e-12)

    def test_resource_time_respects_critical_path(self):
        estimate = chapter(26)["runtime_from_tdepth"]
        times = [estimate(10**10, 10, count) for count in (1, 4, 16, 64)]
        self.assertTrue(all(value >= 100000 for value in times))
        self.assertEqual(times, sorted(times, reverse=True))


if __name__ == "__main__":
    unittest.main()
