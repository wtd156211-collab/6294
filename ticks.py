#!/usr/bin/env python3
"""坐标轴刻度生成引擎。

库与命令行共用同一份结果：给定数据最小/最大值、轴长（格）与标签宽度
预算（格），按 README.md 第二节的口径选步长、算刻度与标签。

所有数值按字面文本解析为精确十进制（Fraction），不经过二进制浮点；
同一份输入跑两遍，刻度序列和标签完全一样。
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from html import escape
from pathlib import Path

STEP_MANTISSAS = (1, 2, 5)  # 候选步长族 s = m × 10^k
CELL_PX = 10  # 页面上 1 格对应的像素数（仅缩放，不影响刻度）


class CliError(Exception):
    """命令行用法或输入不合法。"""


# ---------------------------------------------------------------- 精确十进制基础

def floor_log10(value: Fraction) -> int:
    """floor(log10(value))，value 必须为正，全程整数运算。"""
    if value <= 0:
        raise ValueError("floor_log10 只接受正数")
    num, den = value.numerator, value.denominator
    digits = len(str(num)) - len(str(den))
    if digits >= 0:
        if num < den * 10 ** digits:
            digits -= 1
    else:
        if num * 10 ** (-digits) < den:
            digits -= 1
    return digits


def format_plain(value: Fraction) -> str:
    """精确十进制文本：不带指数、不带多余尾零、不出 -0。"""
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    num, den = abs(value.numerator), value.denominator
    twos = 0
    while den % 2 == 0:
        den //= 2
        twos += 1
    fives = 0
    while den % 5 == 0:
        den //= 5
        fives += 1
    if den != 1:
        raise ValueError(f"不是有限小数: {value}")
    scale = max(twos, fives)
    scaled = num * 2 ** (scale - twos) * 5 ** (scale - fives)
    digits = str(scaled)
    if scale == 0:
        return sign + digits
    if len(digits) > scale:
        text = digits[:-scale] + "." + digits[-scale:]
    else:
        text = "0." + "0" * (scale - len(digits)) + digits
    return sign + text.rstrip("0").rstrip(".")


def format_sci(value: Fraction) -> str:
    """科学计数法标签：尾数e指数，指数不带 + 与前导零。value 非零。"""
    exponent = floor_log10(abs(value))
    mantissa = format_plain(abs(value) / Fraction(10) ** exponent)
    if value < 0:
        mantissa = "-" + mantissa
    return f"{mantissa}e{exponent}"


def use_scientific(lo: Fraction, hi: Fraction) -> bool:
    """按 2.5：满足任一条整轴用科学计数法。"""
    ax = max(abs(lo), abs(hi))
    nonzero = [abs(v) for v in (lo, hi) if v != 0]
    an = min(nonzero) if nonzero else Fraction(0)
    if ax >= 10 ** 6:
        return True
    if 0 < an < Fraction(1, 10 ** 4):
        return True
    if an > 0 and floor_log10(ax) - floor_log10(an) >= 6:
        return True
    return False


def step_candidates(target: Fraction):
    """从族内 >= target 的最小成员起，按 1、2、5 由小到大逐个产出 (m, k)。"""
    k = floor_log10(target)
    start = None
    for m in STEP_MANTISSAS:
        if m * Fraction(10) ** k >= target:
            start = m
            break
    if start is None:
        start, k = 1, k + 1
    m = start
    while True:
        yield m, k
        if m == 1:
            m = 2
        elif m == 2:
            m = 5
        else:
            m, k = 1, k + 1


# ---------------------------------------------------------------- 引擎

@dataclass(frozen=True)
class Tick:
    value: Fraction      # 刻度值（精确十进制）
    text: str            # 刻度文本（定点、无尾零）
    label: str           # 标签文本（定点模式同刻度文本，否则科学计数法）
    width: int           # 标签宽度（字符数，即格数）
    position: Fraction   # 轴上位置（格），i*L/(n-1)


@dataclass(frozen=True)
class TickResult:
    step: Fraction
    step_text: str
    count: int
    scientific: bool
    axis_length: int
    max_label_width: int
    lo: Fraction  # 有效量程下界（零跨度已展开）
    hi: Fraction  # 有效量程上界
    ticks: tuple  # tuple[Tick, ...]

    @property
    def tick_texts(self):
        return [t.text for t in self.ticks]

    @property
    def labels(self):
        return [t.label for t in self.ticks]


def _as_fraction(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ValueError("布尔值不是合法数值")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(Decimal(value))
    raise ValueError(f"不支持的数值类型: {type(value).__name__}")


def solve(min_value, max_value, axis_length: int, max_label_width: int) -> TickResult:
    """按 README 2.3/2.4 选步长并展开刻度。非法输入抛 ValueError。"""
    lo = _as_fraction(min_value)
    hi = _as_fraction(max_value)
    if lo > hi:
        raise ValueError("min 不能大于 max")
    if lo == hi:  # 2.6 零跨度：换成 [v-u, v+u]
        u = Fraction(1) if lo == 0 else Fraction(10) ** floor_log10(abs(lo))
        lo -= u
        hi += u
    n_max = axis_length // (max_label_width + 1) + 1
    if n_max < 2:
        raise ValueError("标签宽度预算过大，连两枚刻度都放不下")
    target = (hi - lo) / (n_max - 1)  # 2.4 起点下界 span/(n_max-1)
    sci = use_scientific(lo, hi)
    for m, k in step_candidates(target):
        step = m * Fraction(10) ** k
        first = math.floor(lo / step)  # 首末向外取整，盖住量程
        last = math.ceil(hi / step)
        count = last - first + 1
        if count < 2 or count > n_max:  # 不变量一：条数
            continue
        values = [Fraction(first + i) * step for i in range(count)]
        if sci:
            labels = ["0" if v == 0 else format_sci(v) for v in values]
            step_text = format_sci(step)
        else:
            labels = [format_plain(v) for v in values]
            step_text = format_plain(step)
        widths = [len(label) for label in labels]
        if any(w > axis_length for w in widths):  # 不变量三：单标签不超轴长
            continue
        if any(  # 不变量三：相邻标签不重叠（按实际宽度）
            2 * axis_length < (count - 1) * (widths[i] + widths[i + 1])
            for i in range(count - 1)
        ):
            continue
        ticks = tuple(
            Tick(
                value=value,
                text=format_plain(value),
                label=label,
                width=width,
                position=Fraction(i * axis_length, count - 1),
            )
            for i, (value, label, width) in enumerate(zip(values, labels, widths))
        )
        return TickResult(
            step=step,
            step_text=step_text,
            count=count,
            scientific=sci,
            axis_length=axis_length,
            max_label_width=max_label_width,
            lo=lo,
            hi=hi,
            ticks=ticks,
        )
    raise RuntimeError("步长候选耗尽（按 2.4 不会发生）")


def result_to_dict(result: TickResult) -> dict:
    """4.3 的 solve 输出。"""
    return {
        "step": result.step_text,
        "count": result.count,
        "ticks": result.tick_texts,
        "labels": result.labels,
    }


# ---------------------------------------------------------------- 输入

def load_range(path) -> dict:
    """读 4.1 的量程 JSON；不合法抛 CliError。"""
    src = Path(path)
    if not src.is_file():
        raise CliError(f"文件不存在: {path}")
    try:
        raw = json.loads(src.read_text(encoding="utf-8"), parse_float=Decimal)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError(f"量程文件读入失败: {exc}") from exc
    if not isinstance(raw, dict):
        raise CliError("量程文件必须是 JSON 对象")
    for key in ("min", "max", "axis_length", "max_label_width"):
        if key not in raw:
            raise CliError(f"缺少字段: {key}")

    def as_number(key):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
            raise CliError(f"{key} 必须是十进制数")
        return Fraction(value)

    def as_int(key):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise CliError(f"{key} 必须是整数")
        return value

    min_v, max_v = as_number("min"), as_number("max")
    axis_length = as_int("axis_length")
    max_label_width = as_int("max_label_width")
    if axis_length < 2:
        raise CliError("axis_length 必须是 ≥ 2 的整数")
    if not 8 <= max_label_width <= axis_length:
        raise CliError("max_label_width 必须满足 8 ≤ max_label_width ≤ axis_length")
    if min_v > max_v:
        raise CliError("min 不能大于 max")
    return {
        "min_value": min_v,
        "max_value": max_v,
        "axis_length": axis_length,
        "max_label_width": max_label_width,
    }


# ---------------------------------------------------------------- 页面

def _px(value: Fraction) -> str:
    """格换算成像素文本，仅用于页面布局。"""
    text = f"{float(value) * CELL_PX:.4f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def render_page(result: TickResult) -> str:
    """按引擎输出画自包含页面：轴、刻度、按实际宽度摆放的标签。"""
    axis_px = result.axis_length * CELL_PX
    boxes = []  # (x0, x1) 像素，供静态标红与页面脚本复核
    parts = [f'<line class="axis-line" x1="0" y1="30" x2="{axis_px}" y2="30"/>']
    for tick in result.ticks:
        x = float(tick.position) * CELL_PX
        half = tick.width * CELL_PX / 2
        boxes.append((x - half, x + half))
        parts.append(
            f'<line class="tick-line" x1="{_px(tick.position)}" y1="25"'
            f' x2="{_px(tick.position)}" y2="35"/>'
        )
    red = set()
    for i in range(result.count - 1):
        if boxes[i][1] > boxes[i + 1][0]:
            red.add(i)
            red.add(i + 1)
    for i, tick in enumerate(result.ticks):
        x0, x1 = boxes[i]
        cls = "tick overlap" if i in red else "tick"
        parts.append(
            f'<g class="{cls}" data-x0="{x0:.4f}" data-x1="{x1:.4f}">'
            f'<rect class="label-box" x="{x0:.4f}" y="42"'
            f' width="{tick.width * CELL_PX}" height="22"/>'
            f'<text class="tick-label" x="{(x0 + x1) / 2:.4f}" y="57"'
            f' text-anchor="middle" textLength="{tick.width * CELL_PX}"'
            f' lengthAdjust="spacingAndGlyphs">{escape(tick.label)}</text></g>'
        )
    svg = "\n".join(parts)
    lines = "\n".join(escape(f"{t.text} {t.label}") for t in result.ticks)
    mode = "科学计数法" if result.scientific else "定点"
    overlap_note = f"相邻标签重叠 {len(red) // 2} 处" + ("（已标红）" if red else "")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>轴刻度核对 step={escape(result.step_text)} count={result.count}</title>
<style>
body {{ font-family: monospace; margin: 24px; color: #222; }}
#axis-summary {{ font-size: 16px; font-weight: bold; }}
#overlap-note {{ margin: 6px 0 16px; }}
#axis-chart {{ display: block; overflow: visible; }}
.axis-line {{ stroke: #333; stroke-width: 1.5; }}
.tick-line {{ stroke: #333; stroke-width: 1; }}
.label-box {{ fill: #f6f6f6; stroke: #bbb; stroke-width: 0.5; }}
.tick-label {{ font-family: monospace; font-size: 14px; fill: #222; }}
.tick.overlap .label-box {{ fill: #ffdada; stroke: #c00000; }}
.tick.overlap .tick-label {{ fill: #c00000; }}
#ticks {{ background: #f6f6f6; padding: 12px; }}
</style>
</head>
<body>
<header id="axis-summary">step={escape(result.step_text)} count={result.count} axis={result.axis_length}格 模式={mode}</header>
<p id="overlap-note">{overlap_note}</p>
<svg id="axis-chart" width="{axis_px + 80}" height="110" viewBox="-40 0 {axis_px + 80} 110" role="img" aria-label="坐标轴刻度">
{svg}
</svg>
<pre id="ticks">{lines}
</pre>
<script type="module">
const groups = [...document.querySelectorAll("#axis-chart g.tick")];
let overlaps = 0;
for (let i = 0; i + 1 < groups.length; i += 1) {{
  const left = groups[i], right = groups[i + 1];
  if (Number(left.dataset.x1) > Number(right.dataset.x0)) {{
    left.classList.add("overlap");
    right.classList.add("overlap");
    overlaps += 1;
  }}
}}
document.getElementById("overlap-note").textContent =
  `相邻标签重叠 ${{overlaps}} 处${{overlaps ? "（已标红）" : ""}}`;
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- 命令行

def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(args) != 3 or args[0] not in ("solve", "page"):
            raise CliError(
                "用法: python ticks.py solve <量程.json> <结果.json>"
                " | python ticks.py page <量程.json> <页面.html>"
            )
        mode, src, dst = args
        spec = load_range(src)
        result = solve(**spec)
        if mode == "solve":
            text = json.dumps(result_to_dict(result), ensure_ascii=False) + "\n"
        else:
            text = render_page(result)
        Path(dst).write_text(text, encoding="utf-8", newline="\n")
    except (CliError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
