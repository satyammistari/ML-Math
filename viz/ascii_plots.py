"""viz/ascii_plots.py — Pure ASCII/Unicode terminal visualisations, zero dependencies."""

import os
import math
from typing import List, Sequence, Tuple, Optional

from config import BOX, BAR_FULL, BAR_EMPTY, SPARKS
from ui.colors import cyan, yellow, green, grey, white, bold_cyan


def _tw() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


# ── 2-D scatter / line plot ───────────────────────────────────────────────────
def scatter(xs: Sequence[float], ys: Sequence[float],
            title: str = "", width: int = 60, height: int = 20,
            x_label: str = "x", y_label: str = "y",
            char: str = "·", color_fn=yellow) -> None:
    """
    ASCII scatter plot on a grid.
    Multiple series can be drawn by calling repeatedly on the same grid (future).
    """
    if not xs:
        return
    xs, ys = list(xs), list(ys)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_rng = x_max - x_min or 1
    y_rng = y_max - y_min or 1

    grid = [[" " for _ in range(width)] for _ in range(height)]

    for x, y in zip(xs, ys):
        col = int((x - x_min) / x_rng * (width  - 1))
        row = int((y - y_min) / y_rng * (height - 1))
        row = height - 1 - row   # flip so y-axis goes up
        col = max(0, min(width-1, col))
        row = max(0, min(height-1, row))
        grid[row][col] = char

    if title:
        print(bold_cyan(f"  {title}"))

    b = BOX
    top_line = b["tl"] + b["h"] * (width + 2) + b["tr"]
    bot_line = b["bl"] + b["h"] * (width + 2) + b["br"]
    print(cyan(top_line))
    for i, row in enumerate(grid):
        if i == 0:
            y_val = y_max
        elif i == height - 1:
            y_val = y_min
        else:
            y_val = y_max - i * y_rng / (height - 1)
        label = grey(f"{y_val:6.3g} ")
        print(label + cyan(b["v"]) + " " + color_fn("".join(row)) + " " + cyan(b["v"]))
    print(cyan(bot_line))
    # x-axis labels
    x_lo = grey(f"       {x_min:.3g}")
    x_hi = grey(f"{x_max:.3g}")
    gap  = width - len(x_lo) - len(x_hi) + 10
    print(x_lo + " " * max(gap, 4) + x_hi)
    print(grey(f"       {'↑ ' + y_label:<20}{'→ ' + x_label}"))


def line_plot(xs: Sequence[float], ys: Sequence[float],
              title: str = "", width: int = 60, height: int = 18,
              char: str = "*", color_fn=green) -> None:
    """ASCII line plot — sort by x then connect nearest points."""
    pairs = sorted(zip(xs, ys))
    scatter([p[0] for p in pairs], [p[1] for p in pairs],
            title=title, width=width, height=height, char=char, color_fn=color_fn)


# ── Multi-series line plot ─────────────────────────────────────────────────────
def multi_line(series: List[Tuple[Sequence[float], Sequence[float], str, str]],
               title: str = "", width: int = 60, height: int = 20) -> None:
    """
    series: list of (xs, ys, label, char)
    """
    all_x = [x for (xs, _, _, _) in series for x in xs]
    all_y = [y for (_, ys, _, _) in series for y in ys]
    if not all_x:
        return
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_rng = x_max - x_min or 1
    y_rng = y_max - y_min or 1
    grid  = [[" " for _ in range(width)] for _ in range(height)]

    chars = ["·", "×", "+", "○", "◆", "▲", "▼"]
    for si, (xs, ys, label, char) in enumerate(series):
        c = char or chars[si % len(chars)]
        for x, y in zip(xs, ys):
            col = int((x - x_min) / x_rng * (width  - 1))
            row = int((y - y_min) / y_rng * (height - 1))
            row = height - 1 - row
            col = max(0, min(width-1, col)); row = max(0, min(height-1, row))
            grid[row][col] = c

    if title:
        print(bold_cyan(f"  {title}"))
    b = BOX
    print(cyan(b["tl"] + b["h"] * (width + 2) + b["tr"]))
    for row in grid:
        print(cyan(b["v"]) + " " + "".join(row) + " " + cyan(b["v"]))
    print(cyan(b["bl"] + b["h"] * (width + 2) + b["br"]))
    # legend
    for si, (_, _, lbl, char) in enumerate(series):
        print(f"  {chars[si % len(chars)]}  {white(lbl)}")


# ── Computation graph ─────────────────────────────────────────────────────────
def comp_graph(nodes: List[Tuple[str, str]],
               edges: List[Tuple[int, int, str]]) -> None:
    """
    nodes: [(id, label), ...]
    edges: [(from_id, to_id, label), ...]
    Draws a simple left-to-right ASCII computation graph.
    """
    b = BOX
    print()
    # Simple linear layout
    labels = {nid: lbl for nid, lbl in nodes}
    node_w = max(len(l) for l in labels.values()) + 4
    line1, line2, line3 = [], [], []
    for nid, lbl in nodes:
        box_top = b["tl"] + b["h"] * (node_w - 2) + b["tr"]
        box_mid = b["v"]  + lbl.center(node_w - 2) + b["v"]
        box_bot = b["bl"] + b["h"] * (node_w - 2) + b["br"]
        line1.append(cyan(box_top))
        line2.append(cyan(box_mid))
        line3.append(cyan(box_bot))

    arrow = yellow(f"  {b['arr_r']}  ")
    print(arrow.join(line1))
    print(arrow.join(line2))
    print(arrow.join(line3))


# ── Heatmap (ASCII) ───────────────────────────────────────────────────────────
def heatmap(matrix, title: str = "", row_labels=None,
            col_labels=None) -> None:
    """ASCII heatmap using block chars based on value intensity."""
    import numpy as np
    M   = np.asarray(matrix, float)
    mn  = M.min(); mx = M.max(); rng = mx - mn or 1
    # character intensity levels
    levels = " ░▒▓█" if True else " .:-=+*#%@"
    n_lev  = len(levels)

    if title:
        print(bold_cyan(f"  {title}"))
    b = BOX

    nrow, ncol = M.shape
    col_w = 5

    # Column headers
    if col_labels:
        hdr = "     " + "".join(str(c).center(col_w) for c in col_labels)
        print(grey(hdr))

    for i in range(nrow):
        row_lbl = grey(str(row_labels[i]).rjust(4) + " ") if row_labels else "      "
        cells   = ""
        for j in range(ncol):
            intensity = int((M[i, j] - mn) / rng * (n_lev - 1))
            ch = levels[intensity]
            cells += yellow(ch * col_w)
        print(row_lbl + cells)


# ── GridWorld map ─────────────────────────────────────────────────────────────
def gridworld(grid: List[List[str]], title: str = "",
              value_fn=None, policy=None) -> None:
    """
    Draw a GridWorld.
    grid cell values: 'S'=start 'G'=goal 'W'=wall '.'=empty
    value_fn: optional 2D array of state values to display
    policy: optional 2D array of arrows ('↑','↓','←','→')
    """
    b = BOX
    if title:
        print(bold_cyan(f"  {title}"))

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    cell_w = 7

    h_sep = b["h"] * cell_w
    top = b["tl"] + (b["tm"] + h_sep) * (cols - 1) + b["tm"] + h_sep + b["tr"]
    mid = b["ml"] + (b["mm"] + h_sep) * (cols - 1) + b["mm"] + h_sep + b["mr"]
    bot = b["bl"] + (b["bm"] + h_sep) * (cols - 1) + b["bm"] + h_sep + b["br"]

    print(cyan(top))
    for i, row in enumerate(grid):
        cells = []
        for j, cell in enumerate(row):
            if cell == "W":
                inner = yellow("  ███  ")
            elif cell == "G":
                inner = green("  [G]  ")
            elif cell == "S":
                inner = cyan("  [S]  ")
            else:
                if value_fn is not None and policy is not None:
                    v = value_fn[i][j]
                    p = policy[i][j]
                    inner = f" {p}{v:+.1f} "
                    inner = inner.center(cell_w)
                    inner = yellow(inner)
                elif value_fn is not None:
                    v = value_fn[i][j]
                    inner = f"{v:+.2f}".center(cell_w)
                    inner = yellow(inner)
                elif policy is not None:
                    p = policy[i][j]
                    inner = p.center(cell_w)
                    inner = green(inner)
                else:
                    inner = "  ...  "
            cells.append(cyan(b["v"]) + inner)
        print("".join(cells) + cyan(b["v"]))
        if i < rows - 1:
            print(cyan(mid))
    print(cyan(bot))


# ── Neural network diagram ────────────────────────────────────────────────────
def neural_net_diagram(layer_sizes: List[int],
                       layer_names: Optional[List[str]] = None) -> None:
    """Draw a simple ASCII neural network architecture."""
    b  = BOX
    n  = len(layer_sizes)
    names = layer_names or [f"L{i}" for i in range(n)]
    max_nodes = max(layer_sizes)
    node_rows = max_nodes * 2 - 1

    node_lines: List[List[str]] = [[] for _ in range(node_rows)]

    for li, (size, name) in enumerate(zip(layer_sizes, names)):
        pad = (node_rows - (size * 2 - 1)) // 2
        col = []
        for r in range(node_rows):
            if r < pad or r >= pad + size * 2 - 1:
                col.append("   ")
            elif (r - pad) % 2 == 0:
                col.append(cyan(" ○ "))
            else:
                col.append("   ")
        for r, c in zip(range(node_rows), col):
            node_lines[r].append(c)
            if li < n - 1:
                node_lines[r].append(grey("──"))

    print()
    for r, parts in enumerate(node_lines):
        print("  " + "".join(parts))
    print()
    # Layer labels
    label_line = "  "
    for i, (size, name) in enumerate(zip(layer_sizes, names)):
        lbl = f"  {name}({size})"
        label_line += bold_cyan(lbl.center(5))
        if i < n - 1:
            label_line += "    "
    print(label_line)
    print()
