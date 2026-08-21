#!/usr/bin/env python
"""Render the batch-sweep curves as self-contained SVG.

No plotting dependency is added: the project environment is pinned by
``uv.lock`` and a chart is not worth perturbing a reproducibility contract for.
These are hand-emitted SVG paths, which also keeps the output diffable and
renderable straight from the repository.

Design follows the project's data-viz method:

* the data's job is change-over-a-continuum for a few named series, so the form
  is a line chart with a log-scaled batch axis;
* latency and throughput are different measures on different scales, so they
  get **separate charts** rather than a dual axis;
* categorical hues are taken in fixed slot order from the reference palette
  (slot 1 blue, slot 2 orange, slot 3 aqua), which are the documented adjacent
  pairs clearing every gate in both modes;
* series are direct-labelled at the right edge as well as colour-coded, so
  identity never rests on colour alone;
* both light and dark surfaces are emitted from the same steps.

Usage::

    python scripts/plot_batch_sweep.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Reference palette, categorical slots in fixed order. (light, dark).
SERIES_COLOURS = {
    "dcnv2_reduced": ("#2a78d6", "#3987e5"),
    "dcdn_reduced": ("#eb6834", "#d95926"),
    "dcnv2_capacity_reduced": ("#1baf7a", "#199e70"),
}
SERIES_LABELS = {
    "dcnv2_reduced": "DCNv2",
    "dcdn_reduced": "DCDN (fused)",
    "dcnv2_capacity_reduced": "DCNv2 capacity-matched",
}
SERIES_ORDER = ("dcnv2_reduced", "dcdn_reduced", "dcnv2_capacity_reduced")

SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
INK_PRIMARY = {"light": "#0b0b0b", "dark": "#ffffff"}
INK_SECONDARY = {"light": "#52514e", "dark": "#c3c2b7"}
GRID = {"light": "#e6e5e1", "dark": "#33332f"}
RULE = {"light": "#c9c8c3", "dark": "#4a4a45"}

WIDTH, HEIGHT = 800, 420
#: Minimum vertical gap between direct labels, in px.
LABEL_MIN_GAP = 15.0
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 76, 206, 56, 56


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _log_ticks(low: float, high: float) -> list[float]:
    ticks: list[float] = []
    exponent = math.floor(math.log10(low))
    while 10.0**exponent <= high * 1.0000001:
        for multiple in (1.0, 2.0, 5.0):
            value = multiple * 10.0**exponent
            if low * 0.999 <= value <= high * 1.001:
                ticks.append(value)
        exponent += 1
    return ticks or [low, high]


def _format_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


class Chart:
    """One line chart: log x, log or linear y, direct-labelled series."""

    def __init__(
        self,
        title: str,
        subtitle: str,
        y_label: str,
        series: dict[str, list[tuple[float, float]]],
        *,
        log_y: bool,
        y_format,
        reference_lines: list[tuple[float, str]] | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.y_label = y_label
        self.series = series
        self.log_y = log_y
        self.y_format = y_format
        self.reference_lines = reference_lines or []
        self.labels = labels or SERIES_LABELS

    def _bounds(self) -> tuple[float, float, float, float]:
        xs = [x for points in self.series.values() for x, _ in points]
        ys = [y for points in self.series.values() for _, y in points]
        ys += [value for value, _ in self.reference_lines]
        low_y, high_y = min(ys), max(ys)
        if self.log_y:
            low_y, high_y = low_y * 0.75, high_y * 1.35
        else:
            span = (high_y - low_y) or 1.0
            low_y, high_y = low_y - span * 0.22, high_y + span * 0.22
        return min(xs), max(xs), low_y, high_y

    def render(self, mode: str) -> str:
        surface = SURFACE[mode]
        primary, secondary = INK_PRIMARY[mode], INK_SECONDARY[mode]
        grid, rule = GRID[mode], RULE[mode]
        x_low, x_high, y_low, y_high = self._bounds()
        plot_w = WIDTH - PAD_LEFT - PAD_RIGHT
        plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM

        def px(value: float) -> float:
            span = math.log10(x_high) - math.log10(x_low)
            return PAD_LEFT + (math.log10(value) - math.log10(x_low)) / span * plot_w

        def py(value: float) -> float:
            if self.log_y:
                span = math.log10(y_high) - math.log10(y_low)
                frac = (math.log10(value) - math.log10(y_low)) / span
            else:
                frac = (value - y_low) / (y_high - y_low)
            return PAD_TOP + (1.0 - frac) * plot_h

        out: list[str] = []
        out.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'font-family="system-ui,-apple-system,Segoe UI,Roboto,sans-serif" '
            f'role="img" aria-label="{_escape(self.title)}">'
        )
        out.append(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{surface}"/>')
        out.append(
            f'<text x="{PAD_LEFT}" y="27" font-size="15" font-weight="600" '
            f'fill="{primary}">{_escape(self.title)}</text>'
        )
        out.append(
            f'<text x="{PAD_LEFT}" y="45" font-size="11.5" fill="{secondary}">'
            f"{_escape(self.subtitle)}</text>"
        )

        for tick in [2.0**k for k in range(0, 15) if x_low <= 2.0**k <= x_high]:
            out.append(
                f'<line x1="{px(tick):.1f}" y1="{PAD_TOP}" x2="{px(tick):.1f}" '
                f'y2="{PAD_TOP + plot_h}" stroke="{grid}" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{px(tick):.1f}" y="{PAD_TOP + plot_h + 18}" '
                f'font-size="10.5" fill="{secondary}" text-anchor="middle">'
                f"{_format_count(tick)}</text>"
            )

        y_ticks = (
            _log_ticks(y_low, y_high)
            if self.log_y
            else [y_low + (y_high - y_low) * k / 5 for k in range(6)]
        )
        for tick in y_ticks:
            out.append(
                f'<line x1="{PAD_LEFT}" y1="{py(tick):.1f}" '
                f'x2="{PAD_LEFT + plot_w}" y2="{py(tick):.1f}" '
                f'stroke="{grid}" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{PAD_LEFT - 9}" y="{py(tick) + 3.5:.1f}" font-size="10.5" '
                f'fill="{secondary}" text-anchor="end">{self.y_format(tick)}</text>'
            )
        out.append(
            f'<line x1="{PAD_LEFT}" y1="{PAD_TOP + plot_h}" '
            f'x2="{PAD_LEFT + plot_w}" y2="{PAD_TOP + plot_h}" '
            f'stroke="{rule}" stroke-width="1"/>'
        )

        for value, label in self.reference_lines:
            out.append(
                f'<line x1="{PAD_LEFT}" y1="{py(value):.1f}" '
                f'x2="{PAD_LEFT + plot_w}" y2="{py(value):.1f}" stroke="{rule}" '
                f'stroke-width="1.5" stroke-dasharray="5 4"/>'
            )
            out.append(
                f'<text x="{PAD_LEFT + 6}" y="{py(value) - 6:.1f}" font-size="10" '
                f'fill="{secondary}">{_escape(label)}</text>'
            )

        drawn = [k for k in SERIES_ORDER if k in self.series]
        for key in drawn:
            colour = SERIES_COLOURS[key][0 if mode == "light" else 1]
            points = self.series[key]
            path = " ".join(
                f"{'M' if index == 0 else 'L'}{px(x):.1f},{py(y):.1f}"
                for index, (x, y) in enumerate(points)
            )
            out.append(
                f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2" '
                f'stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for x, y in points:
                out.append(
                    f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.2" '
                    f'fill="{colour}" stroke="{surface}" stroke-width="2"/>'
                )

        # Direct labels can land on top of each other where series converge.
        # Push them apart along y, keeping their order, and draw a leader line
        # back to the true endpoint so the nudge cannot mislead.
        anchors = sorted(
            ((py(self.series[k][-1][1]), k) for k in drawn), key=lambda item: item[0]
        )
        placed: list[tuple[float, float, str]] = []
        for true_y, key in anchors:
            label_y = true_y
            if placed and label_y - placed[-1][1] < LABEL_MIN_GAP:
                label_y = placed[-1][1] + LABEL_MIN_GAP
            placed.append((true_y, label_y, key))
        for true_y, label_y, key in placed:
            colour = SERIES_COLOURS[key][0 if mode == "light" else 1]
            end_x = px(self.series[key][-1][0])
            if abs(label_y - true_y) > 0.5:
                out.append(
                    f'<path d="M{end_x + 4:.1f},{true_y:.1f} '
                    f'L{end_x + 12:.1f},{label_y:.1f}" fill="none" stroke="{colour}" '
                    f'stroke-width="1" opacity="0.55"/>'
                )
            out.append(
                f'<rect x="{end_x + 12:.1f}" y="{label_y - 4:.1f}" width="8" '
                f'height="8" rx="2" fill="{colour}"/>'
            )
            out.append(
                f'<text x="{end_x + 25:.1f}" y="{label_y + 4:.1f}" '
                f'font-size="11.5" fill="{primary}">'
                f"{_escape(self.labels[key])}</text>"
            )

        out.append(
            f'<text x="{PAD_LEFT}" y="{HEIGHT - 14}" font-size="11" '
            f'fill="{secondary}">batch size (log scale)</text>'
        )
        out.append(
            f'<text transform="translate(18,{PAD_TOP + plot_h / 2}) rotate(-90)" '
            f'font-size="11" fill="{secondary}" text-anchor="middle">'
            f"{_escape(self.y_label)}</text>"
        )
        out.append("</svg>")
        return "\n".join(out)


def build_charts(result: dict[str, Any]) -> dict[str, Chart]:
    measurements = result["measurements"]
    batches = [int(b) for b in result["batches"]]

    def points(arm: str, field: str) -> list[tuple[float, float]]:
        return [(float(b), measurements[arm][str(b)][field]) for b in batches]

    arms = [a for a in SERIES_ORDER if a in measurements]
    subtitle = (
        f"{result['geometry']} geometry, one A100, best implementation per arm; "
        "median of 3 rounds x 300 synchronized forwards"
    )
    return {
        "throughput": Chart(
            "Throughput across the production batch range",
            subtitle,
            "examples / second",
            {a: points(a, "throughput_examples_per_second") for a in arms},
            log_y=True,
            y_format=_format_count,
        ),
        "latency": Chart(
            "Batch latency stays flat: the model never becomes compute-bound",
            subtitle,
            "milliseconds / call (p50)",
            {a: points(a, "batch_latency_ms_p50") for a in arms},
            log_y=False,
            y_format=lambda v: f"{v:.2f}",
        ),
        "ratio": Chart(
            "DCDN relative to DCNv2 across the batch range",
            "runtime-matched requires latency <= 1.10x and throughput >= 0.90x",
            "ratio (DCDN / DCNv2)",
            {
                "dcnv2_reduced": [
                    (float(e["batch"]), e["latency_ratio_p50"])
                    for e in result["dcdn_over_dcnv2"]
                ],
                "dcdn_reduced": [
                    (float(e["batch"]), e["throughput_ratio"])
                    for e in result["dcdn_over_dcnv2"]
                ],
            },
            log_y=False,
            y_format=lambda v: f"{v:.2f}",
            reference_lines=[
                (1.10, "latency threshold 1.10x"),
                (1.00, "parity"),
                (0.90, "throughput threshold 0.90x"),
            ],
            labels={
                "dcnv2_reduced": "latency ratio",
                "dcdn_reduced": "throughput ratio",
            },
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "efficiency"
        / "rq2_batch_sweep_v1"
        / "results_v1.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "docs" / "figures"
    )
    args = parser.parse_args(argv)

    result = json.loads(args.results.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, chart in build_charts(result).items():
        for mode in ("light", "dark"):
            suffix = "" if mode == "light" else "-dark"
            path = args.output_dir / f"batch-sweep-{name}{suffix}.svg"
            path.write_text(chart.render(mode) + "\n", encoding="utf-8")
            print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
