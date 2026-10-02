"""刻度引擎自测：只读 samples/**，输出写临时目录。"""

import json
import os
from decimal import Decimal
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ticks

RANGES_DIR = ROOT / "samples" / "ranges"
EXPECTED_DIR = ROOT / "samples" / "expected"
RANGE_FILES = sorted(RANGES_DIR.glob("*.json"))


def expected_for(range_path):
    lines = (EXPECTED_DIR / (range_path.stem + ".txt")).read_text(
        encoding="utf-8"
    ).splitlines()
    step = lines[0][len("step "):]
    count = int(lines[1][len("count "):])
    pairs = [line.split(" ", 1) for line in lines[2:]]
    return step, count, pairs


def solve_sample(range_path):
    spec = ticks.load_range(range_path)
    return ticks.solve(**spec)


def run_cli(*args, env=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "ticks.py"), *args],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )


class TestExpected(unittest.TestCase):
    def test_all_samples_match_expected(self):
        self.assertTrue(RANGE_FILES, "samples/ranges 为空")
        for range_path in RANGE_FILES:
            with self.subTest(sample=range_path.name):
                step, count, pairs = expected_for(range_path)
                result = solve_sample(range_path)
                self.assertEqual(result.step_text, step)
                self.assertEqual(result.count, count)
                self.assertEqual(result.tick_texts, [p[0] for p in pairs])
                self.assertEqual(result.labels, [p[1] for p in pairs])


class TestInvariants(unittest.TestCase):
    def test_three_invariants_and_step_family(self):
        for range_path in RANGE_FILES:
            with self.subTest(sample=range_path.name):
                spec = ticks.load_range(range_path)
                result = ticks.solve(**spec)
                length = spec["axis_length"]
                n_max = length // (spec["max_label_width"] + 1) + 1
                # 不变量一：条数
                self.assertGreaterEqual(result.count, 2)
                self.assertLessEqual(result.count, n_max)
                # 不变量二：首末刻度盖住有效量程
                self.assertLessEqual(result.ticks[0].value, result.lo)
                self.assertGreaterEqual(result.ticks[-1].value, result.hi)
                # 不变量三：单标签不超轴长、相邻不重叠（按实际宽度）
                widths = [t.width for t in result.ticks]
                for width in widths:
                    self.assertLessEqual(width, length)
                for i in range(result.count - 1):
                    self.assertGreaterEqual(
                        2 * length,
                        (result.count - 1) * (widths[i] + widths[i + 1]),
                    )
                # 步长落在 1/2/5 × 10^k 族里
                k = ticks.floor_log10(result.step)
                self.assertIn(result.step / Fraction(10) ** k, (1, 2, 5))
                # 每枚刻度值是步长的整数倍
                for tick in result.ticks:
                    self.assertEqual((tick.value / result.step).denominator, 1)
                # 标签宽度就是字符数；位置按 2.1 均布
                for i, tick in enumerate(result.ticks):
                    self.assertEqual(tick.width, len(tick.label))
                    self.assertEqual(
                        tick.position, Fraction(i * length, result.count - 1)
                    )
                # 不出 -0 与浮点尾巴：刻度文本精确往返还原
                for tick in result.ticks:
                    self.assertNotEqual(tick.text, "-0")
                    self.assertNotEqual(tick.label, "-0")
                    self.assertEqual(Fraction(Decimal(tick.text)), tick.value)


class TestSpecialCases(unittest.TestCase):
    def test_zero_crossing_has_zero_tick(self):
        result = ticks.solve("-42", "58", 300, 36)
        self.assertIn(Fraction(0), [t.value for t in result.ticks])

    def test_zero_span_expands_around_value(self):
        result = ticks.solve("7.5", "7.5", 400, 40)
        self.assertEqual(result.lo, Fraction("6.5"))
        self.assertEqual(result.hi, Fraction("8.5"))

    def test_zero_span_at_origin(self):
        result = ticks.solve(0, 0, 300, 30)
        self.assertEqual(result.lo, Fraction(-1))
        self.assertEqual(result.hi, Fraction(1))
        self.assertIn(Fraction(0), [t.value for t in result.ticks])

    def test_scientific_for_tiny_range(self):
        result = ticks.solve("0.0000012", "0.0000034", 320, 32)
        self.assertTrue(result.scientific)
        self.assertEqual(result.step_text, "5e-7")
        self.assertEqual(result.labels[0], "1e-6")
        self.assertEqual(result.labels[-1], "3.5e-6")

    def test_scientific_for_multi_magnitude(self):
        result = ticks.solve("0.002", "300000", 480, 44)
        self.assertTrue(result.scientific)
        self.assertEqual(result.labels[0], "0")  # 0 不写指数

    def test_fixed_point_for_plain_range(self):
        result = ticks.solve(0, 100, 480, 44)
        self.assertFalse(result.scientific)
        self.assertEqual(result.step_text, "10")

    def test_determinism_same_input_twice(self):
        for range_path in RANGE_FILES:
            with self.subTest(sample=range_path.name):
                self.assertEqual(solve_sample(range_path), solve_sample(range_path))


class TestFormatHelpers(unittest.TestCase):
    def test_format_plain(self):
        cases = {
            Fraction(0): "0",
            Fraction(-20): "-20",
            Fraction(1, 2): "0.5",
            Fraction(-1, 5): "-0.2",
            Fraction(1, 10 ** 18): "0.000000000000000001",
            Fraction(5, 10 ** 19): "0.0000000000000000005",
            Fraction(3 * 10 ** 11): "300000000000",
            Fraction("1.0000000000000000010"): "1.000000000000000001",
        }
        for value, want in cases.items():
            with self.subTest(value=value):
                self.assertEqual(ticks.format_plain(value), want)

    def test_floor_log10(self):
        cases = {
            Fraction(1): 0, Fraction(9): 0, Fraction(10): 1,
            Fraction(99): 1, Fraction(100): 2,
            Fraction(1, 10): -1, Fraction(1, 100): -2,
            Fraction("0.5"): -1, Fraction("0.002"): -3,
            Fraction("7.5"): 0, Fraction(10) ** 300: 300,
            Fraction(1, 10 ** 300): -300,
        }
        for value, want in cases.items():
            with self.subTest(value=value):
                self.assertEqual(ticks.floor_log10(value), want)

    def test_format_sci(self):
        cases = {
            Fraction(5, 10 ** 7): "5e-7",
            Fraction(15, 10 ** 7): "1.5e-6",
            Fraction(3 * 10 ** 11): "3e11",
            Fraction(-25, 10 ** 6): "-2.5e-5",
            Fraction(1): "1e0",
        }
        for value, want in cases.items():
            with self.subTest(value=value):
                self.assertEqual(ticks.format_sci(value), want)


class TestCli(unittest.TestCase):
    def test_solve_cli_matches_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for range_path in RANGE_FILES:
                with self.subTest(sample=range_path.name):
                    out = Path(tmp) / "out.json"
                    proc = run_cli("solve", str(range_path), str(out))
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    step, count, pairs = expected_for(range_path)
                    data = json.loads(out.read_text(encoding="utf-8"))
                    self.assertEqual(
                        data,
                        {"step": step, "count": count,
                         "ticks": [p[0] for p in pairs],
                         "labels": [p[1] for p in pairs]},
                    )

    def test_page_cli_matches_expected_and_self_contained(self):
        with tempfile.TemporaryDirectory() as tmp:
            for range_path in RANGE_FILES:
                with self.subTest(sample=range_path.name):
                    out = Path(tmp) / "out.html"
                    proc = run_cli("page", str(range_path), str(out))
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    html = out.read_text(encoding="utf-8")
                    step, count, pairs = expected_for(range_path)
                    pre = html.split('<pre id="ticks">', 1)[1].split("</pre>", 1)[0]
                    self.assertEqual(
                        pre, "".join(f"{a} {b}\n" for a, b in pairs)
                    )
                    summary = html.split('id="axis-summary"', 1)[1].split("</header>", 1)[0]
                    self.assertIn(f"step={step}", summary)
                    self.assertIn(f"count={count}", summary)
                    for banned in ("http://", "https://", "src=", "<link", "fetch("):
                        self.assertNotIn(banned, html)

    def test_cli_errors_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bad_json = tmp / "bad.json"
            bad_json.write_text("{", encoding="utf-8")
            missing_key = tmp / "missing.json"
            missing_key.write_text('{"min": 0, "max": 1, "axis_length": 10}',
                                   encoding="utf-8")
            bad_width = tmp / "width.json"
            bad_width.write_text(
                '{"min": 0, "max": 1, "axis_length": 10, "max_label_width": 5}',
                encoding="utf-8")
            reversed_range = tmp / "reversed.json"
            reversed_range.write_text(
                '{"min": 2, "max": 1, "axis_length": 100, "max_label_width": 8}',
                encoding="utf-8")
            out = tmp / "out.json"
            cases = [
                (),
                ("solve",),
                ("solve", str(tmp / "不存在.json"), str(out)),
                ("solve", str(bad_json), str(out)),
                ("solve", str(missing_key), str(out)),
                ("solve", str(bad_width), str(out)),
                ("solve", str(reversed_range), str(out)),
                ("bogus", str(bad_json), str(out)),
            ]
            for args in cases:
                with self.subTest(args=args):
                    proc = run_cli(*args)
                    self.assertEqual(proc.returncode, 2)
                    self.assertTrue(proc.stderr.startswith("error:"), proc.stderr)
                    self.assertFalse(out.exists())

    def test_determinism_across_hash_seeds(self):
        sample = str(RANGES_DIR / "03-zero-crossing.json")
        solve_outputs, page_outputs = [], []
        with tempfile.TemporaryDirectory() as tmp:
            for seed in ("0", "1", "2"):
                env = dict(os.environ, PYTHONHASHSEED=seed)
                out = Path(tmp) / f"out-{seed}.json"
                proc = run_cli("solve", sample, str(out), env=env)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                solve_outputs.append(out.read_bytes())
                page = Path(tmp) / f"page-{seed}.html"
                proc = run_cli("page", sample, str(page), env=env)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                page_outputs.append(page.read_bytes())
        self.assertEqual(solve_outputs[0], solve_outputs[1])
        self.assertEqual(solve_outputs[1], solve_outputs[2])
        self.assertEqual(page_outputs[0], page_outputs[1])
        self.assertEqual(page_outputs[1], page_outputs[2])


if __name__ == "__main__":
    unittest.main()
