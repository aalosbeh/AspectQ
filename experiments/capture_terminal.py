"""
Run a shell command, record its output (with ANSI colours) and render it as a
terminal-window image for the paper's appendix.

    python experiments/capture_terminal.py --name screen_tests \
        --title "AspectQ test suite" -- python -m pytest -q --color=yes

Writes figures/<name>.svg (via rich) and figures/<name>.png (rasterised with
headless Chrome when it is available) plus results/<name>.log with the raw
output.  The command really runs; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time

from rich.console import Console
from rich.terminal_theme import TerminalTheme
from rich.text import Text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

LIGHT = TerminalTheme(
    (252, 252, 251), (11, 11, 11),
    [(11, 11, 11), (197, 48, 48), (34, 120, 60), (160, 110, 0), (42, 120, 214),
     (140, 60, 160), (20, 130, 140), (200, 200, 200)],
    [(90, 90, 90), (230, 73, 72), (27, 175, 122), (237, 161, 0), (57, 135, 229),
     (232, 123, 164), (25, 158, 112), (255, 255, 255)])


def render_png(svg_path: str) -> str:
    """Rasterise a terminal SVG with headless Chrome (2x device scale)."""
    png_path = svg_path[:-4] + ".png"
    if not os.path.exists(CHROME):
        return png_path
    svg = open(svg_path).read()
    w, h = (float(v) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    html = svg_path[:-4] + ".html"
    with open(html, "w") as f:
        f.write("<html><body style='margin:0;background:#ffffff'>" + svg + "</body></html>")
    profile = os.path.join(ROOT, "results", ".chrome-profile")
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--user-data-dir={profile}", "--no-first-run",
                        "--force-device-scale-factor=2",
                        f"--window-size={int(w) + 2},{int(h) + 2}",
                        f"--screenshot={png_path}", "file://" + html],
                       capture_output=True, timeout=180)
    finally:
        os.remove(html)
    return png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", default="Terminal")
    ap.add_argument("--width", type=int, default=104)
    ap.add_argument("--tail", type=int, default=0, help="keep only the last N lines")
    ap.add_argument("--display", default=None, help="command text to show at the prompt")
    ap.add_argument("--render-only", action="store_true",
                    help="only rasterise an existing figures/<name>.svg")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.render_only:
        svg_path = os.path.join(ROOT, "figures", f"{args.name}.svg")
        render_png(svg_path)
        print(f"{args.name}: rendered {svg_path[:-4]}.png")
        return
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    env = dict(os.environ, FORCE_COLOR="1", COLUMNS=str(args.width))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - t0
    raw = proc.stdout + proc.stderr
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"{args.name}.log"), "w") as f:
        f.write(raw)
    lines = raw.rstrip("\n").splitlines()
    if args.tail and len(lines) > args.tail:
        lines = ["..."] + lines[-args.tail:]
    console = Console(record=True, width=args.width, force_terminal=True,
                      color_system="truecolor", file=open(os.devnull, "w"))
    shown = args.display or " ".join(shlex.quote(c) for c in cmd)
    prompt = Text("aspectq % ", style="bold #1c5cab")
    prompt.append(shown, style="bold #0b0b0b")
    console.print(prompt)
    console.print(Text.from_ansi("\n".join(lines)))
    console.print(Text(f"[exit code {proc.returncode}, {elapsed:.1f} s]", style="#52514e"))
    svg_path = os.path.join(ROOT, "figures", f"{args.name}.svg")
    console.save_svg(svg_path, title=args.title, theme=LIGHT)
    png_path = render_png(svg_path)
    print(f"{args.name}: exit {proc.returncode}, {len(lines)} lines, {elapsed:.1f} s -> {png_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
