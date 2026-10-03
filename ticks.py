#!/usr/bin/env python3
"""坐标轴刻度生成引擎：库与命令行共用同一份结果。

口径见 README.md：
- 步长族 s = m x 10^k（m ∈ {1, 2, 5}，k 为整数），从 s0 起由小到大逐个试；
- 三条不变量：条数 2 ≤ n ≤ n_max、首末刻度盖住有效量程、相邻标签不重叠；
- 数值全程精确十进制（Decimal + 整数运算），不经过二进制浮点；
- 零跨度先把有效范围换成 [v-u, v+u]，跨零时 0 必为刻度；
- 标签按 2.5 的阈值决定定点或科学计数法。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from decimal import Decimal
from html import escape
from pathlib import Path

FAMILY_MANTISSAS = (1, 2, 5)
_MAX_CANDIDATES = 10000


class TickError(Exception):
    """输入不合法或找不到满足不变量的步长。"""


@dataclass(frozen=True)
class Tick:
    value: Decimal
    text: str       # 刻度文本（精确十进制、无指数无尾零）
    label: str      # 标签文本（定点或科学计数法）
    width: int      # 标签宽度（字符数 = 格数）
    position: float # 轴上位置（格）


@dataclass(frozen=True)
class Solution:
    step: Decimal
    step_text: str
    scientific: bool
    lo: Decimal
    hi: Decimal
    axis_length: int
    max_label_width: int
    ticks: tuple

    @property
    def count(self) -> int:
        return len(self.ticks)

    def to_result(self) -> dict:
        return {
            "step": self.step_text,
            "count": self.count,
            "ticks": [t.text for t in self.ticks],
            "labels": [t.label for t in self.ticks],
        }


# ---------------------------------------------------------------- 精确十进制基础

def _parts(d: Decimal) -> tuple:
    """返回 (带符号系数 int, 指数)，使 d == coeff * 10**exp。"""
    t = d.as_tuple()
    coeff = 0
    for digit in t.digits:
        coeff = coeff * 10 + digit
    return (-coeff if t.sign else coeff), t.exponent


def _scaled(coeff: int, exp: int) -> Decimal:
    """coeff x 10^exp 的精确 Decimal，不经上下文、不舍入。"""
    if coeff == 0:
        return Decimal(0)
    sign = 1 if coeff < 0 else 0
    digits = tuple(int(c) for c in str(abs(coeff)))
    return Decimal((sign, digits, exp))


def _floor_div(d: Decimal, m: int, k: int) -> int:
    """floor(d / (m x 10^k))，纯整数运算。"""
    coeff, e = _parts(d)
    shift = e - k
    if shift >= 0:
        return (coeff * 10 ** shift) // m
    return coeff // (m * 10 ** (-shift))


def _ceil_div(d: Decimal, m: int, k: int) -> int:
    """ceil(d / (m x 10^k))。"""
    return -_floor_div(-d, m, k)


def _fixed_text(d: Decimal) -> str:
    """精确十进制文本：无指数、无多余尾零、不出 -0。"""
    if d == 0:
        return "0"
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _sci_text(v: Decimal) -> str:
    """非零值的科学计数法文本：尾数e指数，尾数去尾零，指数无 + 与前导零。"""
    coeff, e = _parts(v)
    exp_t = v.adjusted()
    mantissa = _scaled(coeff, e - exp_t)
    return f"{_fixed_text(mantissa)}e{exp_t}"


# ---------------------------------------------------------------- 步长族

def _family_from(m0: int, k0: int):
    """从 (m0, k0) 起按族内从小到大枚举步长 (m, k)。"""
    k = k0
    started = False
    while True:
        for m in FAMILY_MANTISSAS:
            if not started and m < m0:
                continue
            started = True
            yield m, k
        k += 1


def _first_candidate(span: Decimal, unit: int) -> tuple:
    """步长族里 >= span/unit 的最小成员（精确比较，不做除法）。"""
    k = span.adjusted() - len(str(unit)) - 1
    for m, kk in _family_from(1, k):
        if _scaled(m, kk) * unit >= span:
            return m, kk
    raise TickError("内部错误：步长族起点枚举失败")


# ---------------------------------------------------------------- 引擎

def _use_scientific(lo: Decimal, hi: Decimal) -> bool:
    ax = max(abs(lo), abs(hi))
    nonzero = [a for a in (abs(lo), abs(hi)) if a != 0]
    an = min(nonzero) if nonzero else Decimal(0)
    if ax >= _scaled(1, 6):
        return True
    if an != 0 and an < _scaled(1, -4):
        return True
    if an != 0 and ax.adjusted() - an.adjusted() >= 6:
        return True
    return False


def compute_ticks(lo: Decimal, hi: Decimal, axis_length: int,
                  max_label_width: int) -> Solution:
    """按 README 口径为有效量程 [lo, hi] 选步长并展开刻度。"""
    lo = Decimal(lo)
    hi = Decimal(hi)
    if lo > hi:
        raise TickError("min 大于 max")
    if axis_length < 2:
        raise TickError("axis_length 必须是 >= 2 的整数")
    if max_label_width < 1:
        raise TickError("max_label_width 必须是正整数")

    if lo == hi:
        u = Decimal(1) if lo == 0 else _scaled(1, lo.copy_abs().adjusted())
        lo, hi = lo - u, hi + u

    span = hi - lo
    n_max = axis_length // (max_label_width + 1) + 1
    if n_max < 2:
        raise TickError("条数上限不足 2，无可行解")
    scientific = _use_scientific(lo, hi)

    m0, k0 = _first_candidate(span, n_max - 1)
    for tried, (m, k) in enumerate(_family_from(m0, k0)):
        if tried >= _MAX_CANDIDATES:
            break
        q0 = _floor_div(lo, m, k)
        qn = _ceil_div(hi, m, k)
        n = qn - q0 + 1
        if n < 2 or n > n_max:
            continue
        values = [_scaled(q * m, k) for q in range(q0, qn + 1)]
        if scientific:
            labels = ["0" if v == 0 else _sci_text(v) for v in values]
        else:
            labels = [_fixed_text(v) for v in values]
        widths = [len(label) for label in labels]
        if any(w > axis_length for w in widths):
            continue
        if any(2 * axis_length < (n - 1) * (widths[i] + widths[i + 1])
               for i in range(n - 1)):
            continue
        step = _scaled(m, k)
        step_text = _sci_text(step) if scientific else _fixed_text(step)
        ticks = tuple(
            Tick(value=v, text=_fixed_text(v), label=label, width=w,
                 position=i * axis_length / (n - 1))
            for i, (v, label, w) in enumerate(zip(values, labels, widths))
        )
        return Solution(step=step, step_text=step_text, scientific=scientific,
                        lo=lo, hi=hi, axis_length=axis_length,
                        max_label_width=max_label_width, ticks=ticks)
    raise TickError("找不到同时满足三条不变量的步长")


# ---------------------------------------------------------------- 输入

def _bad_constant(name: str):
    raise TickError(f"输入含非法数值 {name}")


def _as_decimal(value, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TickError(f"{name} 必须是十进制数")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise TickError(f"{name} 必须是有限十进制数")
        return value
    raise TickError(f"{name} 必须是十进制数")


def _as_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TickError(f"{name} 必须是整数")
    return value


def load_range(path) -> tuple:
    """读入量程 JSON，返回 (lo, hi, axis_length, max_label_width)。"""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise TickError(f"读不到输入文件 {p}：{exc}")
    try:
        rec = json.loads(text, parse_float=Decimal, parse_constant=_bad_constant)
    except TickError:
        raise
    except ValueError as exc:
        raise TickError(f"输入不是合法 JSON：{exc}")
    if not isinstance(rec, dict):
        raise TickError("输入必须是 JSON 对象")
    for key in ("min", "max", "axis_length", "max_label_width"):
        if key not in rec:
            raise TickError(f"缺少字段 {key}")
    lo = _as_decimal(rec["min"], "min")
    hi = _as_decimal(rec["max"], "max")
    axis_length = _as_int(rec["axis_length"], "axis_length")
    max_label_width = _as_int(rec["max_label_width"], "max_label_width")
    if lo > hi:
        raise TickError("min 大于 max")
    if axis_length < 2:
        raise TickError("axis_length 必须是 >= 2 的整数")
    if not (8 <= max_label_width <= axis_length):
        raise TickError("max_label_width 必须满足 8 <= max_label_width <= axis_length")
    return lo, hi, axis_length, max_label_width


# ---------------------------------------------------------------- 页面

_PAGE_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>刻度核对</title>
<style>
body { font-family: ui-monospace, Menlo, Consolas, monospace; margin: 24px; color: #222; }
#axis-summary { font-size: 15px; margin-bottom: 14px; }
#axis { position: relative; height: 64px; margin: 8px 0 16px 8ch; }
#axis .axis-line { position: absolute; left: 0; top: 8px; height: 2px; background: #444; }
#axis .tick-mark { position: absolute; top: 8px; width: 0; height: 8px; border-left: 1px solid #444; }
#axis .tick-label { position: absolute; top: 24px; white-space: pre; text-align: center; font-size: 13px; }
#axis .tick-label.overlap { color: #d00; font-weight: 700; }
h2 { font-size: 14px; margin: 18px 0 6px; }
#ticks { background: #f6f6f6; border: 1px solid #ddd; padding: 10px; display: inline-block; margin: 0; }
</style>
</head>
<body>
"""

_PAGE_SCRIPT_HEAD = """<script type="module">
const DATA = """

_PAGE_SCRIPT_TAIL = """;
const axis = document.getElementById("axis");
const n = DATA.count;
const L = DATA.axisLength;
axis.style.width = L + "ch";
const line = document.createElement("div");
line.className = "axis-line";
line.style.width = L + "ch";
axis.appendChild(line);
const labelEls = [];
for (const t of DATA.ticks) {
  const mark = document.createElement("div");
  mark.className = "tick-mark";
  mark.style.left = t.position + "ch";
  axis.appendChild(mark);
  const lab = document.createElement("div");
  lab.className = "tick-label";
  lab.textContent = t.label;
  lab.style.left = (t.position - t.width / 2) + "ch";
  lab.style.width = t.width + "ch";
  axis.appendChild(lab);
  labelEls.push(lab);
}
for (let i = 0; i + 1 < n; i++) {
  const a = DATA.ticks[i];
  const b = DATA.ticks[i + 1];
  if (2 * L < (n - 1) * (a.width + b.width)) {
    labelEls[i].classList.add("overlap");
    labelEls[i + 1].classList.add("overlap");
  }
}
</script>
</body>
</html>
"""


def render_page(sol: Solution) -> str:
    """由引擎输出渲染自包含页面；页面数字全部来自 sol，不重算。"""
    data = {
        "step": sol.step_text,
        "count": sol.count,
        "axisLength": sol.axis_length,
        "scientific": sol.scientific,
        "range": [_fixed_text(sol.lo), _fixed_text(sol.hi)],
        "ticks": [
            {"value": t.text, "label": t.label, "width": t.width,
             "position": t.position}
            for t in sol.ticks
        ],
    }
    lines = "\n".join(f"{escape(t.text)} {escape(t.label)}" for t in sol.ticks)
    summary = (f"step={escape(sol.step_text)} count={sol.count}"
               f"　量程 [{escape(_fixed_text(sol.lo))}, {escape(_fixed_text(sol.hi))}]"
               f"　轴长 {sol.axis_length} 格")
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    return (
        _PAGE_HEAD
        + f'<div id="axis-summary">{summary}</div>\n'
        + '<div id="axis"></div>\n'
        + '<h2>刻度与标签</h2>\n'
        + f'<pre id="ticks">{lines}</pre>\n'
        + _PAGE_SCRIPT_HEAD + payload + _PAGE_SCRIPT_TAIL
    )


# ---------------------------------------------------------------- 命令行

def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(args) != 3 or args[0] not in ("solve", "page"):
            raise TickError("用法: python ticks.py <solve|page> <量程.json> <结果.json|页面.html>")
        cmd, src, dst = args
        lo, hi, axis_length, max_label_width = load_range(src)
        sol = compute_ticks(lo, hi, axis_length, max_label_width)
        if cmd == "solve":
            out = json.dumps(sol.to_result(), ensure_ascii=False) + "\n"
        else:
            out = render_page(sol)
        dst_path = Path(dst)
        if str(dst_path.parent) not in ("", "."):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(out, encoding="utf-8", newline="\n")
    except TickError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
