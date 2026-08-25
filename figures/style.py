"""One place for the figure style, so the notebook and the perf chart match.

The two data colours were checked with the project's palette validator: worst
adjacent CVD separation is dE 16.2 (protan) and normal-vision separation 30.5,
both comfortably above the 8 / 15 floors. Teal and grey are structural — a
boundary and an unavailable state — not categorical series, and are excluded
from that check by design.
"""

INK        = "#262626"   # titles, primary labels
MUTED      = "#6b6b6b"   # subtitles, axis numbers
GRID       = "#ebebeb"
RULE       = "#d9d9d9"   # the legend divider
SURFACE    = "#ffffff"

PRODUCT    = "#1a73c8"   # blue  — monomials both architectures reach
RATIO      = "#d21f68"   # pink  — monomials only DCDN reaches
ACCENT     = "#e0a020"   # amber — a third series, always directly labelled
BOUNDARY   = "#4a7c6f"   # teal dashed — the reach boundary
UNREACHED  = "#8c8c8c"   # small grey dot — out of reach at this depth

FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"


def head(width, height, title, subtitle=None, x=28, y=34):
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{x}" y="{y}" font-size="17" font-weight="700" fill="{INK}">{title}</text>',
    ]
    if subtitle:
        out.append(f'<text x="{x}" y="{y + 20}" font-size="12" fill="{MUTED}">{subtitle}</text>')
    return out


def legend(x, y, entries, rule_x=None, height=None):
    """Right-hand legend: marker, bold label, grey subtitle — one block each."""
    out = []
    if rule_x is not None:
        out.append(f'<line x1="{rule_x}" y1="{y - 20}" x2="{rule_x}" y2="{y + (height or 300)}" '
                   f'stroke="{RULE}" stroke-width="1"/>')
    cursor = y
    for marker, colour, label, note in entries:
        if marker == "circle":
            out.append(f'<circle cx="{x + 9}" cy="{cursor - 4}" r="9" fill="{colour}"/>')
        elif marker == "square":
            out.append(f'<rect x="{x}" y="{cursor - 13}" width="18" height="18" rx="3" fill="{colour}"/>')
        elif marker == "dot":
            out.append(f'<circle cx="{x + 9}" cy="{cursor - 4}" r="4" fill="{colour}"/>')
        elif marker == "dash":
            out.append(f'<line x1="{x}" y1="{cursor - 4}" x2="{x + 18}" y2="{cursor - 4}" '
                       f'stroke="{colour}" stroke-width="2" stroke-dasharray="5 4"/>')
        elif marker == "line":
            out.append(f'<line x1="{x}" y1="{cursor - 4}" x2="{x + 18}" y2="{cursor - 4}" '
                       f'stroke="{colour}" stroke-width="3"/>')
        out.append(f'<text x="{x + 30}" y="{cursor} " font-size="14" fill="{INK}">{label}</text>')
        if note:
            out.append(f'<text x="{x + 30}" y="{cursor + 18}" font-size="12" fill="{MUTED}">{note}</text>')
        cursor += 46 if note else 32
    return out
