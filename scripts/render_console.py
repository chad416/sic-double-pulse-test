"""Render captured console output (real command output, saved as text) to PNG."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, dst, title = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
lines = src.read_text(encoding="utf-8", errors="replace").expandtabs(4).splitlines()
width = max(60, max(len(x) for x in lines))
fig_w = min(0.075 * width + 0.6, 16)
fig_h = 0.19 * (len(lines) + 2) + 0.5
fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a19")
fig.text(0.012, 1 - 0.18 / fig_h, title, color="#c3c2b7", family="DejaVu Sans", fontsize=9)
for i, line in enumerate(lines):
    color = "#86b6ef" if line.startswith("$") else "#f0efec"
    if "PASSED" in line or "passed" in line or "All checks passed" in line or "(clean)" in line:
        color = "#5fd35f"
    if "passed=False" in line or "FAILED" in line:
        color = "#ec835a"
    fig.text(
        0.012,
        1 - (0.42 + 0.19 * (i + 1)) / fig_h,
        line,
        color=color,
        family="DejaVu Sans Mono",
        fontsize=8.5,
    )
fig.savefig(dst, dpi=130, facecolor=fig.get_facecolor())
