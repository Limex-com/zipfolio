"""Charts in the QuantStats Pro look: one figure per chart, restyled, returned as SVG."""

from __future__ import annotations

import io
import math
import traceback
from typing import Callable

import numpy as np

#: QuantStats Pro's palette (``quantstats/_plotting/core.py``), so the two
#: reports read as one product: benchmark grey, strategy lime, then the rest.
STRATEGY_COLOR = "#6B9E00"


BENCHMARK_COLOR = "#5C5C5C"


LOSS_COLOR = "#BA516B"


PALETTE = [STRATEGY_COLOR, BENCHMARK_COLOR, LOSS_COLOR, "#4FA487", "#9B59B6",
           "#613F66", "#84B082", "#DC136C", "#559CAD", "#4A5899"]


#: pyfolio's hard-coded colours and what they become here. Each one marks the
#: strategy, the benchmark or a loss in the plots that use it.
#: The single-letter codes ("g", "b", "r") come out in seaborn's colours here:
#: QuantStats Pro calls ``seaborn.set_theme`` on import, which remaps them.
RECOLOR = {
    "#228b22": STRATEGY_COLOR,   # forestgreen: the strategy's cumulative returns
    "#008000": STRATEGY_COLOR,   # green
    "#55a868": STRATEGY_COLOR,   # "g" under seaborn: daily returns, long holdings, means
    "#32cd32": STRATEGY_COLOR,   # limegreen: gross leverage
    "#4682b4": STRATEGY_COLOR,   # steelblue: annual returns, monthly distribution, turnover
    "#4c72b0": STRATEGY_COLOR,   # "b" under seaborn: long round trips
    "#808080": BENCHMARK_COLOR,  # grey: the benchmark
    "#c44e52": LOSS_COLOR,       # "r" under seaborn: short holdings and round trips
    "#ff7f50": LOSS_COLOR,       # coral
    "#fa8072": LOSS_COLOR,       # salmon: the underwater plot
}


#: Lines recoloured to these keep at least this opacity: pyfolio draws its
#: strategy and benchmark curves at 0.6-0.7, which reads as washed out beside
#: QuantStats Pro's solid ones.
SOLID_COLORS = {STRATEGY_COLOR.lower(), BENCHMARK_COLOR.lower()}


SOLID_ALPHA = 0.9


#: The names pyfolio gives the strategy in its legends.
STRATEGY_NAMES = {"Backtest", "Algo", "algo", "Portfolio"}


#: Matplotlib settings for every chart, after QuantStats Pro's seaborn theme.
#: Sizes are for an 8-inch figure shown 620px wide, as QuantStats Pro's are.
CHART_RC = {
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "white",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.bottom": False,
    "axes.spines.left": False,
    "font.size": 11,
    "text.color": "#333333",
    "axes.labelcolor": "#333333",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": "black",
    "axes.titlepad": 12,
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#dddddd",
    "lines.linewidth": 1.5,
}


CHART_WIDTH = 8.0


def chart(
    title: str,
    draw: Callable,
    height: float = 3.0,
    y: str | None = None,
    recolor: dict[str, str] | None = None,
) -> str | None:
    """Draw one chart in its own figure, in the QuantStats Pro look, and return it as SVG.

    ``draw(ax)`` does the plotting -- usually a ``pyfolio.plotting`` function.
    ``y`` is ``"pct"`` for a fraction shown as a percent and ``"growth"`` for
    pyfolio's growth-of-one curves, shown as the return they mean (1.35 is 35%).
    A chart that fails is logged and left out; the section goes on without it.
    """
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(CHART_WIDTH, height))
    try:
        draw(axis)
        _restyle(figure, axis, title, y, {**RECOLOR, **(recolor or {})})
        declutter(figure)
        return _svg(figure)
    except Exception:
        print(f"Warning: the pyfolio chart '{title}' could not be drawn. Continuing without it.")
        print(traceback.format_exc())
        return None
    finally:
        plt.close(figure)


def _restyle(figure, axis, title: str, y: str | None, recolor: dict[str, str]) -> None:
    from matplotlib.collections import QuadMesh

    for ax in figure.axes:
        _recolor(ax, recolor)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        if any(isinstance(collection, QuadMesh) for collection in ax.collections):
            ax.grid(False)
        else:
            ax.grid(True, color="#dddddd", linewidth=0.5)
        if ax.get_xlabel() in {"date", "Date", "index", "dt"}:
            ax.set_xlabel("")
        _restyle_legend(ax, recolor)
    axis.set_title(title)
    if y == "pct":
        percent_axis(axis)
    elif y == "growth":
        percent_axis(axis, offset=1.0)


def percent_axis(axis, offset: float = 0.0) -> None:
    """Label the y axis in percent, with as many decimals as its range needs."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    low, high = axis.get_ylim()
    span = abs(high - low)
    decimals = 0 if span >= 0.2 else 1 if span >= 0.02 else 2
    formatter = FuncFormatter(lambda value, _: f"{(value - offset) * 100:,.{decimals}f}%")
    if axis.get_yscale() == "log":
        # Matplotlib puts a log axis over a few multiples on its minor ticks,
        # every integer of them, and labels the decade below the data too. A
        # few round multiples read better: 0%, 100%, 200%, 400%.
        from matplotlib.ticker import FixedLocator

        candidates = [0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5, 10, 20, 50, 100]
        if high / max(low, 1e-9) > 3:
            candidates = [c for c in candidates if c not in (0.75, 1.25, 1.5)]
        axis.yaxis.set_major_locator(FixedLocator([c for c in candidates if low <= c <= high]))
        axis.yaxis.set_minor_locator(FixedLocator([]))
    axis.yaxis.set_major_formatter(formatter)
    axis.yaxis.set_minor_formatter(NullFormatter())


def _swap(color, recolor: dict[str, str]):
    from matplotlib.colors import to_hex, to_rgba

    try:
        rgba = to_rgba(color)
    except (TypeError, ValueError):
        return None
    replacement = recolor.get(to_hex(rgba, keep_alpha=False))
    return to_rgba(replacement, rgba[3]) if replacement else None


def _recolor(ax, recolor: dict[str, str]) -> None:
    """Move pyfolio's hard-coded colours onto the QuantStats Pro palette, keeping their alpha."""
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import to_hex

    for line in ax.get_lines():
        new = _swap(line.get_color(), recolor)
        if new:
            line.set_color(new)
            if to_hex(new, keep_alpha=False) in SOLID_COLORS and (line.get_alpha() or 1.0) < SOLID_ALPHA:
                line.set_alpha(SOLID_ALPHA)
    for patch in ax.patches:
        new = _swap(patch.get_facecolor(), recolor)
        if new:
            patch.set_facecolor(new)
    for collection in ax.collections:
        if isinstance(collection, QuadMesh):
            continue
        faces = collection.get_facecolor()
        if len(faces):
            new = _swap(faces[0], recolor)
            if new:
                collection.set_facecolor(new)
        edges = collection.get_edgecolor()
        if len(edges):
            new = _swap(edges[0], recolor)
            if new:
                collection.set_edgecolor(new)


def _restyle_legend(ax, recolor: dict[str, str]) -> None:
    """Restyle a legend in place: new colours, ``Strategy`` for pyfolio's ``Backtest``, a light frame."""
    legend = ax.get_legend()
    if legend is None:
        return
    for handle in getattr(legend, "legend_handles", []) or []:
        for getter, setter in (("get_color", "set_color"), ("get_facecolor", "set_facecolor")):
            if hasattr(handle, getter) and hasattr(handle, setter):
                try:
                    current = getattr(handle, getter)()
                    if isinstance(current, np.ndarray) and current.ndim == 2:
                        current = current[0] if len(current) else None
                    new = _swap(current, recolor) if current is not None else None
                    if new:
                        getattr(handle, setter)(new)
                except (TypeError, ValueError):
                    pass
    for text in legend.get_texts():
        if text.get_text() in STRATEGY_NAMES:
            text.set_text("Strategy")
        text.set_fontsize(10)
    frame = legend.get_frame()
    frame.set_edgecolor("#dddddd")
    frame.set_facecolor("white")
    frame.set_alpha(0.9)


def _svg(figure) -> str:
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", pad_inches=0.08)
    text = buffer.getvalue()
    return text[text.index("<svg"):]


def declutter(figure) -> None:
    """Keep pyfolio's text from running together once the figure is on a page.

    pyfolio lays its figures out for a notebook 14 inches wide and leaves text
    density to matplotlib's defaults, and in two places the text collides: the
    monthly-returns heatmap, a third of the width with twelve cells a row, wrote
    "-8.2-4.2" as one string, and dense x axes -- the half-hourly clock of the
    transaction-time histogram, the probability axis of the round-trip section
    -- ran their labels into each other.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        _fit_cell_annotations(axis, renderer)
        _thin_tick_labels(axis, renderer)


def _thin_tick_labels(axis, renderer, gap_px: float = 16.0, gap_share: float = 0.5) -> None:
    """Show every n-th x label when neighbours sit closer than a readable gap.

    The gap is ``gap_px`` or half a label's width, whichever is larger: labels
    that merely do not overlap still read as one string -- "09:30 10:00 10:30"
    on the transaction-time axis had a few pixels between them. Thinning is
    what a reader would do with a pencil: the ticks and the grid stay where
    they are, and the labels left are still horizontal.
    """
    labels = [label for label in axis.get_xticklabels() if label.get_visible() and label.get_text()]
    if len(labels) < 3:
        return
    boxes = sorted(((label.get_window_extent(renderer), label) for label in labels),
                   key=lambda item: item[0].x0)
    widest = max(box.width for box, _ in boxes)
    spacing = min(b.x0 - a.x0 for (a, _), (b, _) in zip(boxes, boxes[1:]))
    needed = widest + max(gap_px, gap_share * widest)
    if spacing <= 0 or needed <= spacing:
        return
    step = math.ceil(needed / spacing)
    for index, (_, label) in enumerate(boxes):
        label.set_visible(index % step == 0)


def _fit_cell_annotations(axis, renderer, fill: float = 0.9) -> None:
    """Make the numbers written into a heatmap's cells fit their cells.

    First the decimals go -- a monthly return to the whole percent is still
    the number a reader takes from a heatmap, and the colour carries the rest --
    and only if that is not enough does the font shrink, the same for every cell.
    """
    texts = [text for text in axis.texts if text.get_visible() and text.get_text()]
    if len(texts) < 2 or not axis.collections:
        return
    columns = len({round(text.get_position()[0], 6) for text in texts})
    cell = axis.get_window_extent(renderer).width / max(columns, 1)

    def widest() -> float:
        return max(text.get_window_extent(renderer).width for text in texts)

    if widest() <= fill * cell:
        return
    for text in texts:
        try:
            text.set_text(str(int(round(float(text.get_text())))))
        except ValueError:
            pass
    if widest() > fill * cell:
        scale = fill * cell / widest()
        for text in texts:
            text.set_fontsize(text.get_fontsize() * scale)


def color_mix(start: str, end: str, share: float) -> str:
    """The colour ``share`` of the way from ``start`` to ``end``."""
    from matplotlib.colors import to_hex, to_rgb

    a, b = np.array(to_rgb(start)), np.array(to_rgb(end))
    return to_hex(a + (b - a) * share)


def held_only(allocation):
    """Allocation with "not held" as a gap in the line rather than a zero.

    A name the strategy does not hold has a weight of exactly zero, and drawn
    as a value that zero is a vertical drop at every exit, a vertical rise at
    every entry, and a band along the axis where every unheld name lies. A
    weight line means something only while the position is open. ``cash`` is
    left as it is.
    """
    names = [column for column in allocation.columns if column != "cash"]
    masked = allocation.copy()
    masked[names] = masked[names].mask(masked[names].eq(0))
    return masked


def mark_isolated(ax, size: float = 12.0) -> None:
    """Dot the points of a gapped line that have no neighbour -- a position held one session.

    A line needs two points to show; without this, a one-session holding would
    vanish from a chart drawn with ``held_only``.
    """
    for line in list(ax.get_lines()):
        values = np.asarray(line.get_ydata(), dtype=float)
        if len(values) < 2:
            continue
        present = np.isfinite(values)
        before = np.concatenate(([False], present[:-1]))
        after = np.concatenate((present[1:], [False]))
        alone = present & ~before & ~after
        if alone.any():
            ax.scatter(np.asarray(line.get_xdata())[alone], values[alone], s=size, color=line.get_color(),
                       zorder=3, linewidths=0)


def anchor_zero(ax) -> None:
    """Keep zero on a weight axis that ``held_only`` gapped.

    With the zeros gone the axis autoscales to the weights actually held and a
    33% position sits at the bottom of the chart, as if it were nothing. Short
    weights, below zero, keep the axis extending below it.
    """
    low, high = ax.get_ylim()
    ax.set_ylim(min(low, 0.0), max(high, 0.0))
