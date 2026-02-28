"""ui/widgets.py — Terminal UI widgets: box, table, panel, sparkline, bar_chart, etc."""

import os
import sys
from typing import List, Optional, Sequence

from config import BOX, BAR_FULL, BAR_EMPTY, SPARKS, TERM_WIDTH
from ui.colors import (bold_cyan, cyan, yellow, green, grey, bold,
                       bold_yellow, header, bold_magenta, white)


def _tw() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return TERM_WIDTH


# ── horizontal rule ────────────────────────────────────────────────────────────
def hr(char: str = "─", color_fn=grey, width: int = 0) -> str:
    w = width or _tw()
    return color_fn(char * w)


# ── box ────────────────────────────────────────────────────────────────────────
def box(title: str = "", lines: List[str] = None, width: int = 0,
        color_fn=cyan, title_color=bold_cyan) -> None:
    w = width or min(_tw(), 100)
    inner = w - 2
    b = BOX
    top = b["tl"] + b["h"] * inner + b["tr"]
    mid = b["ml"] + b["h"] * inner + b["mr"]
    bot = b["bl"] + b["h"] * inner + b["br"]

    print(color_fn(top))
    if title:
        pad = inner - len(title) - 2
        lp, rp = pad // 2, pad - pad // 2
        print(color_fn(b["v"]) + " " * (lp + 1) +
              title_color(title) + " " * (rp + 1) + color_fn(b["v"]))
        print(color_fn(mid))
    if lines:
        for line in lines:
            # strip ANSI for length calc
            import re
            raw = re.sub(r'\033\[[0-9;]*m', '', line)
            pad_r = max(0, inner - 1 - len(raw))
            print(color_fn(b["v"]) + " " + line + " " * pad_r + color_fn(b["v"]))
    print(color_fn(bot))


def box_str(title: str = "", lines: List[str] = None, width: int = 0,
             color_fn=cyan, title_color=bold_cyan) -> str:
    """Return the box as a string instead of printing."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        box(title, lines, width, color_fn, title_color)
    return buf.getvalue()


# ── panel ─────────────────────────────────────────────────────────────────────
def panel(title: str, content: str, color_fn=cyan) -> None:
    lines = content.splitlines()
    box(title, lines, color_fn=color_fn)


# ── table ─────────────────────────────────────────────────────────────────────
def table(headers: List[str], rows: List[List[str]],
          col_colors: Optional[List] = None) -> None:
    if not rows and not headers:
        return
    all_rows = [headers] + rows
    ncols = len(headers)
    widths = [max(len(str(r[i])) for r in all_rows) + 2 for i in range(ncols)]
    b = BOX

    def _row(cells, cfns=None):
        parts = []
        for i, cell in enumerate(cells):
            s = str(cell)
            fn = (cfns[i] if cfns and i < len(cfns) else white)
            parts.append(" " + fn(s.ljust(widths[i] - 2)) + " ")
        return cyan(b["v"]) + cyan(b["v"]).join(parts) + cyan(b["v"])

    def _sep(left, mid_c, right):
        segs = [b["h"] * w for w in widths]
        return cyan(left + mid_c.join(segs) + right)

    print(_sep(b["tl"], b["tm"], b["tr"]))
    print(_row(headers, [bold_cyan] * ncols))
    print(_sep(b["ml"], b["mm"], b["mr"]))
    for row in rows:
        print(_row(row, col_colors))
    print(_sep(b["bl"], b["bm"], b["br"]))


# ── sparkline ─────────────────────────────────────────────────────────────────
def sparkline(data: Sequence[float], label: str = "", color_fn=yellow) -> str:
    if not data:
        return ""
    mn, mx = min(data), max(data)
    rng = mx - mn or 1
    n = len(SPARKS) - 1
    chars = "".join(SPARKS[int((v - mn) / rng * n)] for v in data)
    suffix = f"  min={mn:.3g} max={mx:.3g}"
    prefix = f"{label}: " if label else ""
    return color_fn(prefix + chars) + grey(suffix)


def print_sparkline(data: Sequence[float], label: str = "", color_fn=yellow) -> None:
    print(sparkline(data, label, color_fn))


# ── horizontal bar chart ──────────────────────────────────────────────────────
def bar_chart(labels: List[str], values: List[float],
              title: str = "", width: int = 40,
              color_fn=yellow, label_color=cyan) -> None:
    if title:
        print(bold_cyan(f"  {title}"))
    if not values:
        return
    mx = max(abs(v) for v in values) or 1
    lw = max(len(l) for l in labels)
    for lbl, val in zip(labels, values):
        filled = int(abs(val) / mx * width)
        empty  = width - filled
        bar = BAR_FULL * filled + BAR_EMPTY * empty
        pct = f" {val:8.4g}"
        print(f"  {label_color(lbl.ljust(lw))} {color_fn(bar)}{grey(pct)}")


# ── section header ─────────────────────────────────────────────────────────────
def section_header(text: str) -> None:
    w = _tw()
    left  = BOX["h"] * 3
    right = BOX["h"] * max(0, w - len(text) - 6)
    print(bold_cyan(f"{left} {text} {right}"))


# ── breadcrumb ────────────────────────────────────────────────────────────────
def breadcrumb(*parts: str) -> None:
    sep = grey(f"  {BOX['arr_r']}  ")
    print(sep.join(bold_cyan(p) if i == len(parts)-1 else grey(p)
                   for i, p in enumerate(parts)))
    print()


# ── nav bar ────────────────────────────────────────────────────────────────────
def nav_bar(options: dict) -> None:
    """options = {'b': 'back', 'q': 'quit', ...}"""
    parts = [cyan(f"[{k}]") + grey(f" {v}") for k, v in options.items()]
    line = "  ".join(parts)
    print()
    print(grey("─" * _tw()))
    print("  " + line)
    print(grey("─" * _tw()))


# ── pager ─────────────────────────────────────────────────────────────────────
def pager(lines: List[str], page_size: int = 30) -> None:
    """Display lines with [n]ext / [p]rev paging."""
    total_pages = (len(lines) + page_size - 1) // page_size
    page = 0
    while True:
        start = page * page_size
        chunk = lines[start: start + page_size]
        for ln in chunk:
            print(ln)
        if total_pages > 1:
            print()
            print(grey(f"── Page {page+1}/{total_pages} ──  "
                       f"[n]ext  [p]rev  [q]uit ──"))
            cmd = input("  > ").strip().lower()
            if cmd == "n" and page < total_pages - 1:
                page += 1
            elif cmd == "p" and page > 0:
                page -= 1
            elif cmd == "q":
                break
        else:
            break


# ── code block ────────────────────────────────────────────────────────────────
def code_block(title: str, code: str, lang: str = "python") -> None:
    """Display a syntax-highlighted code block using rich, or plain fallback."""
    try:
        from rich.syntax import Syntax
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        syn = Syntax(code, lang, theme="monokai", line_numbers=True,
                     word_wrap=True)
        console.print(Panel(syn, title=f"[bold cyan]{title}[/bold cyan]",
                            border_style="cyan"))
    except ImportError:
        print(bold_cyan(f"\n┌─ CODE: {title} " + "─" * max(0, 60 - len(title)) + "┐"))
        for i, line in enumerate(code.splitlines(), 1):
            print(f"│ {grey(str(i).rjust(3))}  {line}")
        print(bold_cyan("└" + "─" * 64 + "┘"))


# ── title banner ──────────────────────────────────────────────────────────────
MLMATH_ART = r"""
  ███╗   ███╗██╗      ███╗   ███╗ █████╗ ████████╗██╗  ██╗
  ████╗ ████║██║      ████╗ ████║██╔══██╗╚══██╔══╝██║  ██║
  ██╔████╔██║██║      ██╔████╔██║███████║   ██║   ███████║
  ██║╚██╔╝██║██║      ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║
  ██║ ╚═╝ ██║███████╗ ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║
  ╚═╝     ╚═╝╚══════╝ ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
"""

MLMATH_ART_SIMPLE = r"""
  __  __ _     __  __       _   _
 |  \/  | |   |  \/  | __ _| |_| |__
 | |\/| | |   | |\/| |/ _` | __| '_ \
 | |  | | |___| |  | | (_| | |_| | | |
 |_|  |_|_____|_|  |_|\__,_|\__|_| |_|
"""


def title_banner() -> None:
    """Print the MLMATH ASCII art banner."""
    from config import UNICODE_OK
    art = MLMATH_ART if UNICODE_OK else MLMATH_ART_SIMPLE
    for line in art.splitlines():
        print(bold_magenta(line))
    subtitle = "Mathematics for Machine Learning — Interactive Terminal"
    w = _tw()
    print(bold_cyan(subtitle.center(w)))
    print(grey("─" * w))
