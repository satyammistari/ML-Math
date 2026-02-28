"""ui/menu.py — Main menu and sub-menu rendering + routing system."""

import os
import sys
import json
from typing import List, Callable, Optional, Tuple, Dict

from config import BLOCKS, PROGRESS_FILE, BAR_FULL, BAR_EMPTY
from ui.colors import (bold_cyan, cyan, yellow, green, grey, bold,
                       bold_yellow, bold_magenta, white, hint, red)
from ui.widgets import (title_banner, section_header, breadcrumb,
                        nav_bar, table, bar_chart, hr)


# ── Progress persistence ───────────────────────────────────────────────────────
def _load_progress() -> dict:
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"completed": [], "visited": [], "last_block": None}


def _save_progress(p: dict) -> None:
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(p, f, indent=2)
    except OSError:
        pass


def mark_visited(block_id: str) -> None:
    p = _load_progress()
    if block_id not in p["visited"]:
        p["visited"].append(block_id)
    p["last_block"] = block_id
    _save_progress(p)


def mark_completed(block_id: str) -> None:
    p = _load_progress()
    if block_id not in p["completed"]:
        p["completed"].append(block_id)
    _save_progress(p)


def get_progress() -> dict:
    return _load_progress()


# ── Progress bar string ────────────────────────────────────────────────────────
def _prog_bar(done: int, total: int, width: int = 20) -> str:
    filled = int(done / total * width) if total else 0
    bar = BAR_FULL * filled + BAR_EMPTY * (width - filled)
    return yellow(bar) + grey(f"  {done}/{total}")


# ── Clear screen ───────────────────────────────────────────────────────────────
def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


# ── Main menu ─────────────────────────────────────────────────────────────────
def main_menu() -> Optional[str]:
    """
    Display the main menu. Returns the block ID chosen or None to quit.
    """
    clear()
    title_banner()

    prog  = get_progress()
    done  = len(prog["completed"])
    total = len(BLOCKS)

    print(f"\n  {bold_cyan('Progress:')}  {_prog_bar(done, total)}")
    if prog["last_block"]:
        print(f"  {grey('Last visited:')}  {cyan(prog['last_block'])}")
    print()

    # Block list
    print(bold_cyan("  ┌─ CURRICULUM BLOCKS " + "─" * 52 + "┐"))
    for i, (bid, title, desc) in enumerate(BLOCKS, 1):
        status = green("✓") if bid in prog["completed"] else (
                 cyan("·") if bid in prog["visited"] else grey("○"))
        num    = cyan(f"  {i:2d}.")
        ttl    = bold_cyan(title) if bid in prog["visited"] else white(title)
        dsc    = grey(f"  — {desc}")
        print(f"  {status}  {num}  {ttl}{dsc}")
    print(bold_cyan("  └" + "─" * 73 + "┘"))

    print()
    print(grey("  Enter block number, or:  ") +
          cyan("[/]") + grey(" search  ") +
          cyan("[q]") + grey(" quit") +
          cyan("[r]") + grey(" reset progress"))
    print()
    choice = input(bold_cyan("  mlmath > ")).strip().lower()
    return choice


# ── Block sub-menu ─────────────────────────────────────────────────────────────
TopicEntry = Tuple[str, Callable]   # (display_name, function)

def block_menu(bid: str, bname: str, topics: List[TopicEntry],
               *crumb_parts) -> None:
    """
    Display topics for a block and route to them.
    topics: list of (name, callable) pairs.
    """
    mark_visited(bid)
    while True:
        clear()
        breadcrumb("mlmath", bname)
        print(bold_magenta(f"  {bname}"))
        print()
        for i, (name, _) in enumerate(topics, 1):
            print(f"    {cyan(str(i).rjust(2) + '.')}  {white(name)}")
        print()
        nav_bar({"b": "back to main menu", "q": "quit"})
        choice = input(bold_cyan("  > ")).strip().lower()

        if choice in ("b", "back", ""):
            return
        if choice in ("q", "quit"):
            sys.exit(0)
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(topics):
                try:
                    clear()
                    breadcrumb("mlmath", bname, topics[idx][0])
                    topics[idx][1]()
                except KeyboardInterrupt:
                    print()
                except Exception as exc:
                    print(red(f"\n  Error: {exc}"))
                    import traceback
                    traceback.print_exc()
                input(grey("\n  [Enter] to continue..."))


# ── Topic runner: standard navigation prompt ──────────────────────────────────
def topic_nav(hint_fn: Optional[Callable] = None,
              exercise_fn: Optional[Callable] = None,
              viz_fn: Optional[Callable] = None,
              code_fn: Optional[Callable] = None) -> None:
    """
    Print navigation bar at the end of a topic and handle sub-commands.
    """
    opts: dict = {"b": "back"}
    if hint_fn:      opts["h"] = "hint"
    if viz_fn:       opts["v"] = "visualize"
    if code_fn:      opts["c"] = "show code"
    if exercise_fn:  opts["e"] = "exercise"
    opts["q"] = "quit"

    nav_bar(opts)
    while True:
        cmd = input(bold_cyan("  > ")).strip().lower()
        if cmd in ("b", "back", ""):
            return
        if cmd == "q":
            sys.exit(0)
        if cmd == "h" and hint_fn:
            hint_fn()
        elif cmd == "v" and viz_fn:
            viz_fn()
        elif cmd == "c" and code_fn:
            code_fn()
        elif cmd == "e" and exercise_fn:
            exercise_fn()
        else:
            print(grey("  Unknown command. ") + str(list(opts.keys())))


# ── Keyword search ─────────────────────────────────────────────────────────────
def search_topics(query: str) -> None:
    """Search all block titles and descriptions for a keyword."""
    q = query.lower()
    results = [(i+1, bid, t, d)
               for i, (bid, t, d) in enumerate(BLOCKS)
               if q in t.lower() or q in d.lower()]
    if not results:
        print(grey(f"  No results for '{query}'"))
        return
    print(bold_cyan(f"\n  Search results for '{query}':\n"))
    for num, bid, title, desc in results:
        print(f"    {cyan(str(num).rjust(2) + '.')}  {white(title)}  {grey('— ' + desc)}")
    print()
