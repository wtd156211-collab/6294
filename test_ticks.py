"""刻度引擎的 unittest 自测：只读 samples/**，不写死任何结果。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ticks

ROOT = Path(__file__).resolve().parent
RANGES_DIR = ROOT / "samples" / "ranges"
EXPECTED_DIR = ROOT / "samples" / "expected"
RANGE_FILES = sorted(RANGES_DIR.glob("*.json"))


def expected_of(range_file: Path):
    lines = (EXPECTED_DIR / (range_file.stem + ".txt")).read_text(
        encoding="utf-8").splitlines()
    step = lines[0].split(" ", 1)[1]
    count = int(lines[1].split(" ", 1)[1])
    pairs = [line.split(" ") for line in lines[2:]]
    return step, count, pairs


def solve_file(range_file: Path) -> ticks.Solution:
    return ticks.compute_ticks(*ticks.load_range(range_file))


def int_parts(d: Decimal):
    t = d.as_tuple()
    coeff = 0
    for digit in t.digits:
        coeff = coeff * 10 + digit
    return (-coeff if t.sign else coeff), t.exponent


def effective_range(lo: Decimal, hi: Decimal):
    if lo != hi:
        return lo, hi
    if lo == 0:
        u = Decimal(1)
    else:
        u = Decimal(f"1e{lo.copy_abs().adjusted()}")
    return lo - u, hi + u


class SampleTest(unittest.TestCase):
    """逐样例比对 expected/*.txt（顺序敏感）。"""

    def test_all_samples_match_expected(self):
        self.assertTrue(RANGE_FILES, "samples/ranges 为空")
        for rf in RANGE_FILES:
            with self.subTest(sample=rf.name):
                step, count, pairs = expected_of(rf)
                sol = solve_file(rf)
                self.assertEqual(sol.step_text, step)
                self.assertEqual(sol.count, count)
                self.assertEqual([t.text for t in sol.ticks],
                                 [p[0] for p in pairs])
                self.assertEqual([t.label for t in sol.ticks],
                                 [p[1] for p in pairs])


class InvariantTest(unittest.TestCase):
    """对每份样例独立核对三条不变量与步长族、整数倍、标签宽度。"""

    def check_one(self, rf: Path):
        lo_in, hi_in, axis_length, max_label_width = ticks.load_range(rf)
        sol = ticks.compute_ticks(lo_in, hi_in, axis_length, max_label_width)
        lo, hi = effective_range(lo_in, hi_in)

        # 步长落在 1/2/5 x 10^k 族里
        coeff, _ = int_parts(sol.step)
        coeff = abs(coeff)
        while coeff % 10 == 0:
            coeff //= 10
        self.assertIn(coeff, (1, 2, 5))

        # 每枚刻度都是步长的整数倍（精确十进制）
        _, s_exp = int_parts(sol.step)
        s_coeff = abs(int_parts(sol.step)[0])
        for t in sol.ticks:
            t_coeff, t_exp = int_parts(t.value)
            shift = t_exp - s_exp
            if shift >= 0:
                self.assertEqual((t_coeff * 10 ** shift) % s_coeff, 0)
            else:
                self.assertEqual(t_coeff % (s_coeff * 10 ** (-shift)), 0)

        # 不变量一：条数
        n_max = axis_length // (max_label_width + 1) + 1
        self.assertGreaterEqual(sol.count, 2)
        self.assertLessEqual(sol.count, n_max)

        # 不变量二：首末刻度盖住有效量程
        self.assertLessEqual(sol.ticks[0].value, lo)
        self.assertGreaterEqual(sol.ticks[-1].value, hi)

        # 不变量三：标签宽度与不重叠
        n = sol.count
        for t in sol.ticks:
            self.assertEqual(t.width, len(t.label))
            self.assertLessEqual(t.width, axis_length)
        for a, b in zip(sol.ticks, sol.ticks[1:]):
            self.assertGreaterEqual(2 * axis_length,
                                    (n - 1) * (a.width + b.width))

        # 刻度文本：无指数、无 -0、无浮点尾巴
        for t in sol.ticks:
            self.assertNotIn("e", t.text.lower())
            self.assertNotEqual(t.text, "-0")
            self.assertEqual(Decimal(t.text), t.value)

    def test_invariants_hold(self):
        for rf in RANGE_FILES:
            with self.subTest(sample=rf.name):
                self.check_one(rf)

    def test_zero_crossing_has_zero_tick(self):
        sol = solve_file(RANGES_DIR / "03-zero-crossing.json")
        self.assertIn(Decimal(0), [t.value for t in sol.ticks])

    def test_zero_span_expands_around_value(self):
        sol = solve_file(RANGES_DIR / "06-zero-span.json")
        self.assertLessEqual(sol.ticks[0].value, Decimal("6.5"))
        self.assertGreaterEqual(sol.ticks[-1].value, Decimal("8.5"))
        sol0 = solve_file(RANGES_DIR / "07-zero-span-zero.json")
        self.assertLessEqual(sol0.ticks[0].value, Decimal(-1))
        self.assertGreaterEqual(sol0.ticks[-1].value, Decimal(1))

    def test_scientific_and_fixed_modes(self):
        tiny = solve_file(RANGES_DIR / "04-tiny.json")
        self.assertTrue(tiny.scientific)
        huge = solve_file(RANGES_DIR / "05-huge.json")
        self.assertTrue(huge.scientific)
        plain = solve_file(RANGES_DIR / "01-positive-plain.json")
        self.assertFalse(plain.scientific)
        self.assertEqual([t.text for t in plain.ticks],
                         [t.label for t in plain.ticks])


class DeterminismTest(unittest.TestCase):
    def test_engine_is_deterministic(self):
        for rf in RANGE_FILES:
            with self.subTest(sample=rf.name):
                args = ticks.load_range(rf)
                a = ticks.compute_ticks(*args)
                b = ticks.compute_ticks(*args)
                self.assertEqual(a.to_result(), b.to_result())
                self.assertEqual(ticks.render_page(a), ticks.render_page(b))

    def test_cli_bytes_stable_across_hash_seeds(self):
        rf = RANGES_DIR / "03-zero-crossing.json"
        for cmd, suffix in (("solve", ".json"), ("page", ".html")):
            outputs = []
            for seed in ("0", "1", "2"):
                with tempfile.TemporaryDirectory() as td:
                    out = Path(td) / ("out" + suffix)
                    env = dict(os.environ, PYTHONHASHSEED=seed)
                    proc = subprocess.run(
                        [sys.executable, str(ROOT / "ticks.py"), cmd,
                         str(rf), str(out)],
                        capture_output=True, env=env)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    outputs.append(out.read_bytes())
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(outputs[1], outputs[2])


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "ticks.py"), *args],
                              capture_output=True, text=True)

    def test_solve_output_shape(self):
        rf = RANGES_DIR / "03-zero-crossing.json"
        step, count, pairs = expected_of(rf)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "r.json"
            proc = self.run_cli("solve", str(rf), str(out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            got = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(got["step"], step)
            self.assertEqual(got["count"], count)
            self.assertEqual(got["ticks"], [p[0] for p in pairs])
            self.assertEqual(got["labels"], [p[1] for p in pairs])
            self.assertEqual(len(got["ticks"]), got["count"])

    def test_page_contains_pre_and_summary(self):
        rf = RANGES_DIR / "03-zero-crossing.json"
        step, count, pairs = expected_of(rf)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "p.html"
            proc = self.run_cli("page", str(rf), str(out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            html = out.read_text(encoding="utf-8")
            pre = html.split('<pre id="ticks">', 1)[1].split("</pre>", 1)[0]
            self.assertEqual(pre.splitlines(),
                             [f"{p[0]} {p[1]}" for p in pairs])
            self.assertIn(f"step={step}", html)
            self.assertIn(f"count={count}", html)
            self.assertIn('id="axis-summary"', html)
            self.assertIn('type="module"', html)
            for bad in ("http://", "https://", "<script src", "<link",
                        "fetch("):
                self.assertNotIn(bad, html)

    def test_errors_exit_2_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "x.json"
            cases = [
                ("solve", "/nonexistent/range.json", str(out)),
                ("page", "/nonexistent/range.json", str(out)),
                ("bogus", "a", "b"),
                ("solve", "only-one"),
                (),
            ]
            for args in cases:
                with self.subTest(args=args):
                    proc = self.run_cli(*args)
                    self.assertEqual(proc.returncode, 2)
                    self.assertTrue(proc.stderr.startswith("error:"),
                                    proc.stderr)
            self.assertFalse(out.exists())

    def test_invalid_content_rejected(self):
        bad_payloads = [
            "{}",
            '{"min": 0, "max": 1, "axis_length": 100}',
            '{"min": 5, "max": 1, "axis_length": 100, "max_label_width": 8}',
            '{"min": 0, "max": 1, "axis_length": 1, "max_label_width": 8}',
            '{"min": 0, "max": 1, "axis_length": 100, "max_label_width": 7}',
            '{"min": 0, "max": 1, "axis_length": 100, "max_label_width": 101}',
            '{"min": "0", "max": 1, "axis_length": 100, "max_label_width": 8}',
            "not json",
        ]
        with tempfile.TemporaryDirectory() as td:
            for i, payload in enumerate(bad_payloads):
                with self.subTest(payload=payload):
                    src = Path(td) / f"bad{i}.json"
                    src.write_text(payload, encoding="utf-8")
                    out = Path(td) / f"out{i}.json"
                    proc = self.run_cli("solve", str(src), str(out))
                    self.assertEqual(proc.returncode, 2)
                    self.assertTrue(proc.stderr.startswith("error:"))
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
