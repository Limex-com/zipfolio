"""Charts that respond: every matplotlib chart, once drawn, rebuilt as an Apache ECharts chart.

The report draws each chart with matplotlib -- most of them through pyfolio's own
plotting functions -- and restyles it (``style.chart``). This module then reads
what was drawn off the axes, artist by artist, and writes the same picture as an
ECharts option: lines, bars and histograms, filled areas, scatter points, heatmap
cells, reference lines and spans, with the axes' own limits, ticks and labels.
The page draws it with tooltips on hover, legend entries that switch a series
off and on, and a zoom on every time axis.

Nothing is re-plotted by hand, so a chart keeps looking like its static self, and
a chart that holds something this module does not read -- an image, a second set
of axes -- stays a static SVG (``Unsupported``).

The option is plain JSON; the functions it needs -- tick labels, tooltips, how to
draw a bar or an area -- are attached on the page by ``RUNTIME``.
"""

from __future__ import annotations

import html
import json
import math
import re
from contextvars import ContextVar
from pathlib import Path

import numpy as np

#: Whether ``style.chart`` returns interactive charts; set for one page by ``build_tearsheet``.
INTERACTIVE: ContextVar[bool] = ContextVar("zipfolio_interactive", default=False)

#: The page shows an 8-inch figure 620px wide.
PX_PER_INCH = 620 / 8
PX_PER_POINT = PX_PER_INCH / 72
#: A line of this many points or fewer, without a label, is drawn with its kind as one series.
SHORT_LINE = 6

LIBRARY = Path(__file__).with_name("static") / "echarts.min.js"

#: An artist with this gid is scenery -- the cells of a style box -- drawn without a tooltip.
BACKGROUND = "zipfolio-background"
#: An artist may carry its own tooltips, one per point or cell, as ``artist.zipfolio_tips``:
#: the section knows what a point is better than the axes do.
TIPS = "zipfolio_tips"
#: Axes with this attribute set get a tooltip per point rather than one per position on the x axis:
#: a map of portfolios by risk and return, where a line is a model and the points are the news.
ITEM_TIPS = "zipfolio_item_tips"


class Unsupported(Exception):
    """The chart holds something this module does not convert; it stays a static SVG."""


def library() -> str:
    """ECharts itself (Apache-2.0), inlined into the page so it needs no network."""
    return LIBRARY.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------
# Numbers and colours.
# ----------------------------------------------------------------------------

def _num(value) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(f"{value:.6g}")


def _color(color, alpha: float | None = None) -> str:
    from matplotlib.colors import to_rgba

    r, g, b, a = to_rgba(color)
    if alpha is not None:
        a = a * alpha if a < 1 else alpha
    if a >= 0.999:
        return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"
    return f"rgba({round(r * 255)},{round(g * 255)},{round(b * 255)},{a:.3g})"


def _visible_color(color) -> bool:
    from matplotlib.colors import to_rgba

    try:
        return to_rgba(color)[3] > 0.01
    except (TypeError, ValueError):
        return False


_SIGNED = re.compile(r"^([−\-+]?)([^\d.−\-+]*)([\d,]*\.?\d+)(.*)$")
_PREFIXED = re.compile(r"^([^\d.−\-+]*)([−\-+]?)([\d,]*\.?\d+)(.*)$")


def _parse_label(text: str):
    """``"−12.5%"`` -> (-12.5, prefix "", suffix "%", decimals 1, grouped False); ``None`` if not a number."""
    text = text.strip().replace("−", "-")
    for pattern, sign_first in ((_SIGNED, True), (_PREFIXED, False)):
        match = pattern.match(text)
        if not match:
            continue
        sign, prefix = (match.group(1), match.group(2)) if sign_first else (match.group(2), match.group(1))
        digits, suffix = match.group(3), match.group(4)
        if re.search(r"\d", suffix) or re.search(r"\d", prefix):
            return None
        value = float(digits.replace(",", ""))
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        return (-value if sign == "-" else value), prefix, suffix, decimals, "," in digits
    return None


def _decimals(value: float, labelled: int, extra: int) -> int:
    """Decimals for a value on an axis labelled with ``labelled``: none for a whole number or a large one."""
    if abs(value - round(value)) < 1e-9 or (labelled == 0 and abs(value) >= 100):
        return 0
    return labelled + extra


def _general(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    if abs(value) >= 1e5:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.4g}" if abs(value) >= 1 else f"{value:.3g}"


# ----------------------------------------------------------------------------
# Axes.
# ----------------------------------------------------------------------------

class AxisMap:
    """One axis: its kind, limits, the ticks matplotlib labelled, and how a value on it is written."""

    def __init__(self, axes, which: str):
        from matplotlib import dates as mdates

        self.which = which
        axis = axes.xaxis if which == "x" else axes.yaxis
        locator, formatter = axis.get_major_locator(), axis.get_major_formatter()
        self.time = isinstance(locator, mdates.DateLocator) or isinstance(
            formatter, (mdates.DateFormatter, mdates.AutoDateFormatter, mdates.ConciseDateFormatter))
        self.log = (axes.get_xscale() if which == "x" else axes.get_yscale()) == "log"
        if self.time:
            epoch = np.datetime64(mdates.get_epoch(), "ms").astype("int64")
            self._offset = float(epoch)
        low, high = axes.get_xlim() if which == "x" else axes.get_ylim()
        self.inverse = bool(low > high)
        self.lim = (min(low, high), max(low, high))
        self.name = (axes.get_xlabel() if which == "x" else axes.get_ylabel()) or ""
        labels = axes.get_xticklabels() if which == "x" else axes.get_yticklabels()
        position = 0 if which == "x" else 1
        self.ticks = [(float(label.get_position()[position]), label.get_text(), label.get_visible())
                      for label in labels]
        self.ticks = [(at, text, shown) for at, text, shown in self.ticks
                      if self.lim[0] - 1e-9 <= at <= self.lim[1] + 1e-9]
        self.spec = self._label_spec()

    def out(self, value):
        """A value in the axis's data units, as ECharts gets it: epoch milliseconds on a time axis."""
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        if self.time:
            return int(round(value * 86_400_000 + self._offset))
        return _num(value)

    def _label_spec(self) -> dict | None:
        """How matplotlib wrote this axis's numbers, as ``label = a * value + b``, or ``None`` for words."""
        parsed = [(at, _parse_label(text)) for at, text, _ in self.ticks if text.strip()]
        if len(parsed) < 2 or any(found is None for _, found in parsed):
            return None
        prefixes = {found[1] for _, found in parsed}
        suffixes = {found[2] for _, found in parsed}
        if len(prefixes) > 1 or len(suffixes) > 1:
            return None
        at = np.array([position for position, _ in parsed])
        shown = np.array([found[0] for _, found in parsed])
        if np.ptp(at) == 0:
            return None
        a, b = np.polyfit(at, shown, 1)
        decimals = [found[3] for _, found in parsed]
        tolerance = 0.6 * 10 ** -max(decimals) + 1e-9 * np.abs(shown).max()
        if np.abs(a * at + b - shown).max() > max(tolerance, 1e-12):
            return None
        general = self.log or len(set(decimals)) > 1
        return {"a": float(a), "b": float(b), "d": -1 if general else max(decimals),
                "p": prefixes.pop(), "s": suffixes.pop(), "c": any(found[4] for _, found in parsed)}

    def words(self) -> bool:
        """Whether the ticks are words -- names, regimes -- rather than numbers."""
        return self.spec is None and any(text.strip() for _, text, _ in self.ticks)

    def format(self, value: float, extra: int = 1) -> str:
        """A value as the axis writes it, a decimal finer; a tick's word when the axis is labelled in words."""
        if value is None:
            return "-"
        if self.time:
            from matplotlib import dates as mdates

            return f"{mdates.num2date(value):%-d %b %Y}"
        if self.words():
            nearest = min(self.ticks, key=lambda tick: abs(tick[0] - value), default=None)
            return nearest[1] if nearest is not None and nearest[1].strip() else _general(value)
        spec = self.spec
        if spec is None:
            return _general(value)
        shown = spec["a"] * value + spec["b"]
        if spec["d"] < 0:
            body = _general(abs(shown))
        else:
            body = f"{abs(shown):,.{_decimals(shown, spec['d'], extra)}f}"
            if not spec["c"] and abs(shown) < 1_000:
                body = body.replace(",", "")
        negative = shown < 0 and body.strip("0.,") != ""
        return f"{'-' if negative else ''}{spec['p']}{body}{spec['s']}"

    def option(self) -> dict:
        """The ECharts axis: matplotlib's limits and labelled ticks; dates on a time axis left to ECharts."""
        font = {"color": "#666666", "fontSize": 11, "fontFamily": "Arial"}
        option = {
            "type": "time" if self.time else "log" if self.log else "value",
            "inverse": self.inverse,
            "axisLine": {"show": False}, "axisTick": {"show": False},
            "splitLine": {"show": True, "lineStyle": {"color": "#dddddd", "width": 0.5}},
            "axisLabel": {**font, "hideOverlap": True},
        }
        if self.name:
            option.update({"name": self.name, "nameLocation": "middle",
                           "nameTextStyle": {"color": "#333333", "fontSize": 11, "fontWeight": "bold",
                                             "fontFamily": "Arial"}})
        if self.time:
            option["min"], option["max"] = self.out(self.lim[0]), self.out(self.lim[1])
            return option
        option["min"], option["max"] = _num(self.lim[0]), _num(self.lim[1])
        shown = [(at, text) for at, text, visible in self.ticks if visible and text.strip()]
        if shown:
            option["axisLabel"]["customValues"] = [_num(at) for at, _ in shown]
            option["axisLabel"]["zfLabels"] = [[_num(at), text.replace("−", "-")] for at, text in shown]
            option["splitLine"]["show"] = False
            option["zfGrid"] = [_num(at) for at, _ in shown]
        return option


# ----------------------------------------------------------------------------
# Artists.
# ----------------------------------------------------------------------------

_DASHES = {"--": "dashed", "dashed": "dashed", ":": "dotted", "dotted": "dotted", "-.": "dashed",
           "dashdot": "dashed"}
_SYMBOLS = {"o": "circle", "s": "rect", "D": "diamond", "d": "diamond", "^": "triangle", "v": "triangle",
            ".": "circle", "*": "pin", "p": "roundRect", "h": "roundRect"}
_STRATEGY = {"Backtest", "Algo", "algo", "Portfolio"}


def _label(artist) -> str | None:
    label = artist.get_label()
    if not label or str(label).startswith("_"):
        return None
    return "Strategy" if label in _STRATEGY else str(label)


def _line_style(line) -> dict:
    return {"color": _color(line.get_color(), line.get_alpha()),
            "width": round(max(line.get_linewidth(), 0.4) * PX_PER_POINT, 2),
            "type": _DASHES.get(line.get_linestyle(), "solid")}


class Converter:
    """Reads one set of axes into ECharts series."""

    def __init__(self, axes):
        self.axes = axes
        self.x = AxisMap(axes, "x")
        self.y = AxisMap(axes, "y")
        self.series: list[dict] = []
        self.legend: list[str] = []
        self.marks = {"lines": [], "areas": []}
        self.rects: dict = {}
        self.polys: dict = {}
        self.segments: dict = {}
        self.texts = list(axes.texts)
        self.areas: list[dict] = []
        #: Tooltips a section gave its bars, by patch: ``container.zipfolio_tips``, one per bar.
        self.own_tips: dict[int, str] = {}

    # -- what gets drawn ---------------------------------------------------

    def run(self) -> None:
        axes = self.axes
        if axes.images:
            raise Unsupported("an image")
        legend = axes.get_legend()
        if legend is not None:
            self.legend = ["Strategy" if text.get_text() in _STRATEGY else text.get_text()
                           for text in legend.get_texts()]
        for line in axes.get_lines():
            if line.get_visible():
                self._line(line)
        containers = {}
        for container in axes.containers:
            label = _label(container)
            own = getattr(container, TIPS, None)
            for index, patch in enumerate(getattr(container, "patches", []) or []):
                if label:
                    containers[id(patch)] = label
                if own is not None and index < len(own):
                    self.own_tips[id(patch)] = own[index]
        for patch in axes.patches:
            if patch.get_visible():
                self._patch(patch, containers.get(id(patch)) or _label(patch))
        for collection in axes.collections:
            if collection.get_visible():
                self._collection(collection)
        for text in self.texts:
            self._text(text)
        self._flush()

    def _transform_kind(self, transform) -> str:
        axes = self.axes
        if transform == axes.transData:
            return "data"
        if transform == axes.get_yaxis_transform(which="grid"):
            return "y-only"  # x in axes fractions: a horizontal line or band
        if transform == axes.get_xaxis_transform(which="grid"):
            return "x-only"  # y in axes fractions: a vertical line or band
        if transform == axes.transAxes:
            return "axes"
        return "other"

    def _line(self, line) -> None:
        kind = self._transform_kind(line.get_transform())
        data = np.asarray(line.get_xydata(), dtype=float)
        style = _line_style(line)
        if kind == "y-only" and len(data):
            self.marks["lines"].append({"yAxis": _num(data[0, 1]), "lineStyle": style})
            return
        if kind == "x-only" and len(data):
            self.marks["lines"].append({"xAxis": self.x.out(data[0, 0]), "lineStyle": style})
            return
        if kind != "data" or not len(data):
            return
        name = _label(line)
        marker = line.get_marker()
        markers_only = line.get_linestyle() in ("None", "", " ", "none")
        data = data[np.isfinite(data[:, 0])]
        points = [[self.x.out(x), self.y.out(y) if self.y.out(y) is not None else "-"] for x, y in data]
        if not points:
            return
        if markers_only:
            size = max(line.get_markersize(), 1) * PX_PER_POINT
            symbol = _SYMBOLS.get(marker, "circle")
            symbol_size = size
            if marker in ("|", 2, 3):
                symbol, symbol_size = "rect", [1.4, size * 1.6]
            elif marker in ("_", 0, 1):
                symbol, symbol_size = "rect", [size * 1.6, 1.4]
            face = line.get_markerfacecolor()
            color = _color(face if _visible_color(face) else line.get_color(), line.get_alpha())
            self.series.append({
                "type": "scatter", "name": name or "", "data": points, "symbol": symbol,
                "symbolSize": symbol_size, "itemStyle": {"color": color, "opacity": 1}, "z": 3,
                "zfTips": getattr(line, TIPS, None) or
                [self._point_tip(name, p) if np.isfinite(p[1]) else "" for p in data],
            })
            return
        if name is None and len(points) <= SHORT_LINE:
            key = (style["color"], style["width"], style["type"])
            self.segments.setdefault(key, []).append([[p[0], p[1]] for p in points if p[1] != "-"])
            return
        step = {"steps-post": "end", "steps-pre": "start", "steps-mid": "middle", "steps": "start"}.get(
            line.get_drawstyle())
        series = {
            "type": "line", "name": name or "", "data": points, "showSymbol": False, "symbol": "none",
            "lineStyle": style, "itemStyle": {"color": style["color"]}, "connectNulls": False,
            "z": 2 + int(line.get_zorder()), "emphasis": {"disabled": True},
            "sampling": "lttb" if len(points) > 3000 else None,
        }
        if step:
            series["step"] = step
        if marker not in ("None", "", None, " ", "none"):
            series.update({"showSymbol": True, "symbol": _SYMBOLS.get(marker, "circle"),
                           "symbolSize": max(line.get_markersize(), 2) * PX_PER_POINT})
        self.series.append({k: v for k, v in series.items() if v is not None})

    def _point_tip(self, name, point) -> str:
        head = f"{html.escape(name)}<br>" if name else ""
        return f"{head}{html.escape(self.x.format(point[0]))}: <b>{html.escape(self.y.format(point[1]))}</b>"

    def _patch(self, patch, name: str | None) -> None:
        from matplotlib.patches import Rectangle

        face = patch.get_facecolor()
        if not _visible_color(face) and not (patch.get_fill() is False and _visible_color(patch.get_edgecolor())):
            return
        kind = self._transform_kind(patch.get_data_transform())
        color = _color(face, patch.get_alpha()) if patch.get_fill() else "rgba(0,0,0,0)"
        edge = patch.get_edgecolor()
        stroke = (_color(edge, patch.get_alpha()) if _visible_color(edge) and patch.get_linewidth() > 0 else None)
        if isinstance(patch, Rectangle) and kind in ("x-only", "y-only"):
            x0, y0, w, h = patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()
            area = ([{"xAxis": self.x.out(x0)}, {"xAxis": self.x.out(x0 + w)}] if kind == "x-only" else
                    [{"yAxis": _num(y0)}, {"yAxis": _num(y0 + h)}])
            area[0]["itemStyle"] = {"color": color}
            self.marks["areas"].append(area)
            return
        if kind != "data":
            return
        if isinstance(patch, Rectangle):
            x0, y0, w, h = patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()
            if not all(math.isfinite(v) for v in (x0, y0, w, h)):
                return
            key = BACKGROUND if patch.get_gid() == BACKGROUND else name
            bucket = self.rects.setdefault(key, {"items": [], "colors": [], "tips": [], "stroke": stroke,
                                                 "labels": []})
            bucket["items"].append([self.x.out(x0), self.x.out(x0 + w), self.y.out(y0), self.y.out(y0 + h)])
            bucket["colors"].append(color)
            bucket["tips"].append(self.own_tips.get(id(patch)) or self._rect_tip(name, x0, y0, w, h))
            bucket["labels"].append(None)
            return
        path = patch.get_path().transformed(patch.get_patch_transform())
        for polygon in path.to_polygons():
            self._polygon(name, polygon, color, stroke, patch.get_zorder())

    def _rect_tip(self, name, x0, y0, w, h) -> str:
        head = f"{html.escape(name)}<br>" if name else ""
        if self.y.words() and not self.x.words():
            where, value = self.y.format(y0 + h / 2), self.x.format(w)
        elif self.x.words() or self.x.time:
            where, value = self.x.format(x0 + w / 2), self.y.format(h)
        else:
            where, value = f"{self.x.format(x0)} to {self.x.format(x0 + w)}", self.y.format(h)
        return f"{head}{html.escape(where)}: <b>{html.escape(value)}</b>"

    def _polygon(self, name, polygon, color, stroke, zorder, area=None) -> None:
        points = [[self.x.out(x), self.y.out(y)] for x, y in polygon if math.isfinite(x) and math.isfinite(y)]
        points = [p for p in points if p[0] is not None and p[1] is not None]
        if len(points) < 3:
            return
        key = (name, color, stroke, int(zorder))
        bucket = self.polys.setdefault(key, [])
        bucket.append(points)
        if area is not None and name:
            self.areas.append({"name": name, **area})

    def _collection(self, collection) -> None:
        from matplotlib.collections import LineCollection, PathCollection, PolyCollection, QuadMesh

        if isinstance(collection, QuadMesh):
            self._mesh(collection)
            return
        if self._transform_kind(collection.get_transform()) not in ("data",) and not isinstance(
                collection, PathCollection):
            return
        name = _label(collection)
        alpha = collection.get_alpha()
        faces = collection.get_facecolor()
        if isinstance(collection, PathCollection):
            if self._transform_kind(collection.get_offset_transform()) != "data":
                return
            offsets = np.asarray(collection.get_offsets(), dtype=float)
            if not len(offsets):
                return
            sizes = collection.get_sizes()
            colors = [_color(c, alpha) for c in faces] if len(faces) else ["#333333"]
            data = []
            for i, (x, y) in enumerate(offsets):
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                item = {"value": [self.x.out(x), self.y.out(y)]}
                if len(colors) > 1:
                    item["itemStyle"] = {"color": colors[i % len(colors)]}
                data.append(item)
            size = float(np.sqrt(sizes[0])) * PX_PER_POINT if len(sizes) else 5.0
            own = getattr(collection, TIPS, None)
            finite = [i for i, (x, y) in enumerate(offsets) if math.isfinite(x) and math.isfinite(y)]
            self.series.append({
                "type": "scatter", "name": name or "", "data": data, "symbolSize": max(size, 5),
                "itemStyle": {"color": colors[0], "opacity": 1}, "z": 3 + int(collection.get_zorder()),
                "zfTips": [own[i] for i in finite] if own is not None else
                [self._point_tip(name, offsets[i]) for i in finite],
            })
            return
        if isinstance(collection, LineCollection):
            colors = collection.get_color() if hasattr(collection, "get_color") else collection.get_edgecolor()
            widths = collection.get_linewidth()
            for i, segment in enumerate(collection.get_segments()):
                color = _color(colors[i % len(colors)], alpha) if len(colors) else "#333333"
                width = float(widths[i % len(widths)]) * PX_PER_POINT if len(widths) else 1.0
                key = (color, round(width, 2), "solid")
                self.segments.setdefault(key, []).append(
                    [[self.x.out(x), self.y.out(y)] for x, y in segment if math.isfinite(x) and math.isfinite(y)])
            return
        if isinstance(collection, PolyCollection):
            edges = collection.get_edgecolor()
            widths = collection.get_linewidth()
            for i, path in enumerate(collection.get_paths()):
                face = faces[i % len(faces)] if len(faces) else (0, 0, 0, 0)
                if not _visible_color(face):
                    continue
                stroke = (_color(edges[i % len(edges)], alpha) if len(edges) and len(widths)
                          and float(widths[i % len(widths)]) > 0 and _visible_color(edges[i % len(edges)]) else None)
                for polygon in path.to_polygons():
                    self._polygon(name, polygon, _color(face, alpha), stroke, collection.get_zorder(),
                                  area=_two_curves(polygon))
            return
        raise Unsupported(type(collection).__name__)

    def _mesh(self, mesh) -> None:
        """A heatmap: one rectangle per cell, in its own colour, with the number written in it when there is one."""
        coordinates = mesh.get_coordinates()
        mesh.update_scalarmappable()
        faces = mesh.get_facecolor()
        values = np.ma.filled(np.ma.asarray(mesh.get_array(), dtype=float).ravel(), np.nan)
        rows, columns = coordinates.shape[0] - 1, coordinates.shape[1] - 1
        written = {}
        for text in self.texts:
            x, y = text.get_position()
            written[(round(x, 3), round(y, 3))] = text
        used = set()
        own = getattr(mesh, TIPS, None)
        bucket = self.rects.setdefault("__mesh__", {"items": [], "colors": [], "tips": [], "stroke": "#ffffff",
                                                    "labels": [], "label_colors": []})
        for row in range(rows):
            for column in range(columns):
                index = row * columns + column
                value = values[index] if index < len(values) else np.nan
                if not math.isfinite(value):
                    continue
                (x0, y0), (x1, y1) = coordinates[row, column], coordinates[row + 1, column + 1]
                center = (round((x0 + x1) / 2, 3), round((y0 + y1) / 2, 3))
                text = written.get(center)
                if text is not None:
                    used.add(id(text))
                bucket["items"].append([self.x.out(x0), self.x.out(x1), self.y.out(y0), self.y.out(y1)])
                bucket["colors"].append(_color(faces[index % len(faces)]))
                where = f"{self.y.format((y0 + y1) / 2)} · {self.x.format((x0 + x1) / 2)}"
                shown = text.get_text() if text is not None else f"{value:.4g}"
                bucket["tips"].append(own[index] if own is not None else
                                      f"{html.escape(where)}: <b>{html.escape(shown)}</b>")
                bucket["labels"].append(text.get_text() if text is not None else None)
                bucket["label_colors"].append(_color(text.get_color()) if text is not None else None)
        self.texts = [text for text in self.texts if id(text) not in used]

    def _text(self, text) -> None:
        if not text.get_visible() or not text.get_text().strip():
            return
        if self._transform_kind(text.get_transform()) != "data":
            return
        x, y = text.get_position()
        if self.x.out(x) is None or self.y.out(y) is None:
            return
        # Text that starts or ends at its point -- a name beside its dot -- sits beside it.
        side = {"left": "right", "right": "left"}.get(text.get_horizontalalignment(), "inside")
        self.series.append({
            "type": "scatter", "data": [[self.x.out(x), self.y.out(y)]], "symbolSize": 0, "silent": True,
            "label": {"show": True, "formatter": text.get_text().strip(), "color": _color(text.get_color()),
                      "fontSize": round(text.get_fontsize() * PX_PER_POINT), "fontFamily": "Arial",
                      "fontWeight": "bold" if text.get_fontweight() in ("bold", 700) else "normal",
                      "position": side, "distance": 3},
            "tooltip": {"show": False}, "z": 6,
        })

    # -- into series ---------------------------------------------------------

    def _flush(self) -> None:
        for name, bucket in self.rects.items():
            unnamed = name in ("__mesh__", BACKGROUND) or name is None
            scenery = name == BACKGROUND
            self.series.append({
                "type": "custom", "name": "" if unnamed else name, "data": bucket["items"],
                "encode": {"x": [0, 1], "y": [2, 3]}, "z": 0 if scenery else 1, "silent": scenery,
                "zf": {"kind": "rect", "colors": bucket["colors"], "stroke": bucket["stroke"],
                       "labels": bucket["labels"] if any(bucket["labels"]) else None,
                       "labelColors": bucket.get("label_colors")},
                "zfTips": None if scenery else bucket["tips"],
                "itemStyle": {"color": bucket["colors"][0]},
            })
        for (name, color, stroke, zorder), polygons in self.polys.items():
            extents = []
            for polygon in polygons:
                xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
                extents.append([min(xs), max(xs), min(ys), max(ys)])
            self.series.append({
                "type": "custom", "name": name or "", "data": extents, "encode": {"x": [0, 1], "y": [2, 3]},
                "z": 1 + zorder, "silent": name is None,
                "zf": {"kind": "poly", "polys": polygons, "color": color, "stroke": stroke},
                "zfTips": [html.escape(name)] * len(polygons) if name else None,
                "itemStyle": {"color": color},
            })
        for (color, width, dash), segments in self.segments.items():
            segments = [segment for segment in segments if len(segment) >= 2]
            if not segments:
                continue
            extents = [[min(p[0] for p in s), max(p[0] for p in s), min(p[1] for p in s), max(p[1] for p in s)]
                       for s in segments]
            self.series.append({
                "type": "custom", "name": "", "data": extents, "encode": {"x": [0, 1], "y": [2, 3]}, "z": 3,
                "silent": True,
                "zf": {"kind": "segs", "segs": segments, "color": color, "width": width, "dash": dash},
            })
        if self.marks["lines"] or self.marks["areas"]:
            self.series.append({
                "type": "line", "name": "", "data": [], "silent": True, "z": 0,
                "markLine": {"silent": True, "symbol": "none", "label": {"show": False}, "animation": False,
                             "data": self.marks["lines"]} if self.marks["lines"] else None,
                "markArea": {"silent": True, "data": self.marks["areas"]} if self.marks["areas"] else None,
            })


def _two_curves(polygon) -> dict | None:
    """``fill_between``'s polygon read back into its curves: ``{x, top, bottom}``, or ``None``.

    Matplotlib draws it as the first point of the lower curve, the upper curve
    forward, the last point of the lower curve, and the lower curve backward.
    """
    points = np.asarray(polygon, dtype=float)
    # A closed path repeats its first point at the end; the lower curve's own first point can equal it too.
    if len(points) % 2 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 6 or len(points) % 2:
        return None
    n = (len(points) - 2) // 2
    upper, lower = points[1:n + 1], points[n + 2:][::-1]
    if len(lower) != n or not np.allclose(upper[:, 0], lower[:, 0]):
        return None
    return {"x": upper[:, 0], "top": upper[:, 1], "bottom": lower[:, 1]}


# ----------------------------------------------------------------------------
# The chart.
# ----------------------------------------------------------------------------

def option_for(figure, axes, title: str, height_inches: float) -> tuple[dict, int]:
    """The ECharts option for a drawn figure, and its height in pixels. Raises ``Unsupported``."""
    others = [ax for ax in figure.axes if ax is not axes and ax.get_label() != "<colorbar>"]
    if others:
        raise Unsupported("more than one set of axes")
    converter = Converter(axes)
    converter.run()
    x, y = converter.x, converter.y
    series = [s for s in converter.series if s.get("data") is not None]
    if not any(s.get("data") for s in series):
        raise Unsupported("nothing drawn")
    areas = []
    for area in converter.areas:
        first, second = np.asarray(area["top"], dtype=float), np.asarray(area["bottom"], dtype=float)
        if np.ptp(first) == 0 and np.ptp(second) == 0:
            continue  # a band of constant height: a span, not a quantity
        # fill_between(x, curve, baseline) is the curve over its baseline; a stacked layer, between
        # the layers under and over it, is its own thickness.
        value = first - second if np.ptp(second) == 0 else second - first if np.ptp(first) == 0 else \
            np.abs(first - second)
        areas.append({"name": area["name"], "x": [x.out(v) for v in area["x"]], "value": [_num(v) for v in value]})

    lines = [s for s in series if s["type"] == "line" and s.get("data")]
    if areas and not lines:
        # The pointer of an axis tooltip snaps to a line's points; areas have none. A line no one
        # sees, along the first area's dates, gives it the dates to snap to.
        guide = areas[0]
        series.append({"type": "line", "name": "__axis__", "data": [list(p) for p in zip(guide["x"], guide["value"])],
                       "showSymbol": False, "symbol": "none", "lineStyle": {"width": 0, "opacity": 0}, "z": 0,
                       "emphasis": {"disabled": True}})
        lines = series[-1:]
    by_axis = x.time or (lines and not any(s.get("zf", {}).get("kind") == "rect" for s in series)
                         and not getattr(axes, ITEM_TIPS, False))
    legend = [name for name in converter.legend if any(s.get("name") == name for s in series)]
    top = 34 + (24 if legend else 0)
    # ECharts sizes its margins from ticks of its own choosing, not the labels it is given: the
    # margins are measured here from matplotlib's labels, about 6-7px a character at 11px Arial.
    labels = [sum(7.4 if c.isupper() else 6.2 for c in line)
              for _, text, visible in y.ticks if visible for line in text.split("\n")]
    left = (26 if y.name else 6) + max(labels, default=25) + 8
    rows = max((text.count("\n") + 1 for _, text, visible in x.ticks if visible), default=1)
    bottom = 8 + 14 * rows + (18 if x.name else 0) + (28 if x.time else 0)
    height = int(height_inches * PX_PER_INCH) + (24 if legend else 0) + (28 if x.time else 0)
    option = {
        "animation": False,
        "title": {"text": title, "left": "center", "top": 4,
                  "textStyle": {"fontSize": 14, "fontWeight": "bold", "color": "#000000", "fontFamily": "Arial"}},
        "grid": {"left": round(left), "right": 22, "top": top, "bottom": bottom, "containLabel": False},
        "xAxis": x.option(), "yAxis": y.option(),
        "series": series,
        "tooltip": {"trigger": "axis" if by_axis else "item", "confine": True,
                    "axisPointer": {"type": "line", "lineStyle": {"color": "#999999", "width": 1}},
                    "textStyle": {"fontSize": 12, "fontFamily": "Arial"}},
        # Zoom on dates; enlarging, saving and copying are the page's own buttons, beside it (page._TOOLS).
        **({"toolbox": {"right": 60, "top": 2, "itemSize": 12, "iconStyle": {"borderColor": "#999999"},
                        "feature": {"dataZoom": {"yAxisIndex": "none", "title": {"zoom": "Zoom", "back": "Back"}},
                                    "restore": {"title": "Reset"}}}} if x.time else {}),
        "zf": {"x": x.spec, "y": y.spec, "xTime": x.time, "areas": areas,
               "xWords": [[_num(at), text] for at, text, _ in x.ticks] if x.words() else None,
               "yWords": [[_num(at), text] for at, text, _ in y.ticks] if y.words() else None},
    }
    if x.name:
        option["xAxis"]["nameGap"] = 8 + 14 * rows
    if y.name:
        # The rotated name sits left of the tick labels.
        option["yAxis"]["nameGap"] = round(left - 18)
    if legend:
        option["legend"] = {"data": legend, "top": 26, "type": "scroll", "itemWidth": 16, "itemHeight": 8,
                            "textStyle": {"fontSize": 11, "fontFamily": "Arial", "color": "#333333"}}
    if x.time:
        option["dataZoom"] = [{"type": "slider", "height": 14, "bottom": 6, "showDetail": False,
                               "filterMode": "none", "borderColor": "#dddddd",
                               "fillerColor": "rgba(107,158,0,0.12)", "moveHandleSize": 0,
                               "handleStyle": {"color": "#ffffff", "borderColor": "#999999"},
                               "dataBackground": {"lineStyle": {"color": "#cccccc"},
                                                  "areaStyle": {"color": "#eeeeee"}}}]
    return option, height


def native_chart(option: dict, height: int, title: str) -> str:
    """A chart written as ECharts in the first place -- a treemap has no matplotlib picture to read back."""
    option = {"animation": False, "tooltip": {"trigger": "item", "confine": True,
                                              "textStyle": {"fontSize": 12, "fontFamily": "Arial"}},
              "title": {"text": title, "left": "center", "top": 4,
                        "textStyle": {"fontSize": 14, "fontWeight": "bold", "color": "#000000",
                                      "fontFamily": "Arial"}},
              **option}
    return chart_html(option, height, title)


def _plain(value):
    """numpy's numbers and booleans as JSON's."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON")


def chart_html(option: dict, height: int, title: str) -> str:
    """A chart's place on the page: an empty box the page draws into, and its option beside it."""
    data = json.dumps(option, separators=(",", ":"), allow_nan=False, default=_plain).replace("</", "<\\/")
    return (f'<div class="ichart" role="img" aria-label="{html.escape(title)}" style="height:{height}px">'
            f'<script type="application/json">{data}</script></div>')


#: Turns the JSON options into charts: label formatters, tooltips, and the custom series'
#: drawing. Charts are drawn when they come into view -- the parts of the page start closed.
RUNTIME = r"""
(function () {
  if (!window.echarts) return;
  var charts = [];
  function general(v) {
    if (v == null || isNaN(v)) return "-";
    var a = Math.abs(v);
    if (a >= 1e5 || Number.isInteger(v)) return Math.round(v).toLocaleString("en-US");
    return a >= 1 ? Number(v.toPrecision(4)).toLocaleString("en-US", {maximumFractionDigits: 6}) : String(Number(v.toPrecision(3)));
  }
  function linear(spec, extra) {
    return function (v) {
      if (v == null || v === "-" || isNaN(v)) return "-";
      if (!spec) return general(v);
      var x = spec.a * v + spec.b, body;
      if (spec.d < 0) body = general(Math.abs(x));
      else {
        var whole = Math.abs(x - Math.round(x)) < 1e-9, d = (whole || (spec.d === 0 && Math.abs(x) >= 100)) ? 0 : spec.d + extra;
        body = Math.abs(x).toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d,
          useGrouping: !!spec.c || Math.abs(x) >= 1000});
      }
      var negative = x < 0 && /[1-9]/.test(body);
      return (negative ? "-" : "") + spec.p + body + spec.s;
    };
  }
  var MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  function day(ms) { var d = new Date(ms); return d.getUTCDate() + " " + MONTHS[d.getUTCMonth()] + " " + d.getUTCFullYear(); }
  function words(pairs, exact) {
    // The label of the tick nearest a value; with ``exact``, only of a tick at the value itself.
    return function (v) {
      var best = null, gap = Infinity;
      pairs.forEach(function (pair) { var g = Math.abs(pair[0] - v); if (g < gap) { gap = g; best = pair; } });
      if (!best) return exact ? "" : general(v);
      if (exact && gap > 1e-6 * Math.max(1, Math.abs(best[0]))) return "";
      return best[1];
    };
  }
  function dot(color) { return '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;background:' + color + '"></span>'; }
  function nearest(xs, x) {
    var lo = 0, hi = xs.length - 1;
    if (x < xs[0] || x > xs[hi]) return -1;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
    return (x - xs[lo] <= xs[hi] - x) ? lo : hi;
  }
  function axisLabels(axis) {
    var labels = axis.axisLabel && axis.axisLabel.zfLabels;
    if (labels) axis.axisLabel.formatter = words(labels, true);
    if (axis.zfGrid) {
      axis.splitLine = {show: false};
      axis.axisTick = {show: true, customValues: axis.zfGrid, length: 0};
    }
  }
  function gridLines(option) {
    // matplotlib's grid, at its own ticks: a mark line per tick on the first series.
    var marks = [];
    ["xAxis", "yAxis"].forEach(function (key) {
      (option[key].zfGrid || []).forEach(function (v) {
        var m = {}; m[key] = v; m.lineStyle = {color: "#dddddd", width: 0.5, type: "solid"}; marks.push(m);
      });
    });
    if (!marks.length) return;
    option.series.push({type: "line", data: [], silent: true, z: 0,
      markLine: {silent: true, symbol: "none", label: {show: false}, animation: false, data: marks}});
  }
  function custom(series) {
    var zf = series.zf;
    series.renderItem = function (params, api) {
      var i = params.dataIndex, sys = params.coordSys;
      var clip = {x: sys.x, y: sys.y, width: sys.width, height: sys.height};
      if (zf.kind === "rect") {
        var p0 = api.coord([api.value(0), api.value(2)]), p1 = api.coord([api.value(1), api.value(3)]);
        if (!p0 || !p1 || isNaN(p0[0] + p0[1] + p1[0] + p1[1])) return;
        var shape = echarts.graphic.clipRectByRect({x: Math.min(p0[0], p1[0]), y: Math.min(p0[1], p1[1]),
          width: Math.abs(p1[0] - p0[0]), height: Math.abs(p1[1] - p0[1])}, clip);
        if (!shape) return;
        var rect = {type: "rect", shape: shape, style: {fill: zf.colors[i], stroke: zf.stroke || null,
          lineWidth: zf.stroke ? 0.5 : 0}};
        var text = zf.labels && zf.labels[i];
        if (!text) return rect;
        return {type: "group", children: [rect, {type: "text", style: {text: text, x: shape.x + shape.width / 2,
          y: shape.y + shape.height / 2, align: "center", verticalAlign: "middle",
          fill: (zf.labelColors && zf.labelColors[i]) || "#222", font: "11px Arial"}}]};
      }
      if (zf.kind === "poly") {
        var points = zf.polys[i].map(function (p) { return api.coord(p); });
        points = echarts.graphic.clipPointsByRect(points, clip);
        return {type: "polygon", shape: {points: points}, style: {fill: zf.color, stroke: zf.stroke || null,
          lineWidth: zf.stroke ? 0.8 : 0}};
      }
      if (zf.kind === "segs") {
        var pts = zf.segs[i].map(function (p) { return api.coord(p); });
        pts = echarts.graphic.clipPointsByRect(pts, clip);
        return {type: "polyline", shape: {points: pts}, style: {stroke: zf.color, lineWidth: zf.width,
          lineDash: zf.dash === "solid" ? null : [4, 3], fill: null}};
      }
    };
  }
  function hydrate(option) {
    var zf = option.zf || {};
    var xfmt = zf.xTime ? day : zf.xWords ? words(zf.xWords, false) : linear(zf.x, 1);
    var yfmt = zf.yWords ? words(zf.yWords, false) : linear(zf.y, 1);
    if (option.xAxis && option.yAxis) {  // a treemap has no axes
      axisLabels(option.xAxis); axisLabels(option.yAxis);
      gridLines(option);
    }
    option.series.forEach(function (s) { if (s.type === "custom") custom(s); });
    // A tooltip that explains wraps rather than running off the chart.
    option.tooltip.extraCssText = "max-width:360px;white-space:normal;";
    option.tooltip.formatter = option.tooltip.trigger === "axis" ? function (params) {
      if (!params.length) return "";
      var x = params[0].axisValue, rows = [];
      params.forEach(function (p) {
        if ((p.seriesType !== "line" && p.seriesType !== "scatter") || p.seriesName === "__axis__") return;
        var v = Array.isArray(p.value) ? p.value[1] : p.value;
        if (v == null || v === "-") return;
        rows.push(dot(p.color) + (p.seriesName ? p.seriesName + ": " : "") + "<b>" + yfmt(v) + "</b>");
      });
      (zf.areas || []).forEach(function (a) {
        var i = nearest(a.x, x);
        if (i < 0) return;
        rows.push(dot("#999") + a.name + ": <b>" + yfmt(a.value[i]) + "</b>");
      });
      option.series.forEach(function (s) {
        if (!s.zf || s.zf.kind !== "rect" || !zf.xTime) return;
        s.data.forEach(function (r, i) {
          if (x >= r[0] && x <= r[1]) rows.push(dot(s.zf.colors[i]) + (s.name ? s.name + ": " : "") + "<b>" + yfmt(r[3] - r[2]) + "</b>");
        });
      });
      return "<b>" + xfmt(x) + "</b><br>" + rows.join("<br>");
    } : function (p) {
      var s = option.series[p.seriesIndex];
      if (p.data && p.data.tip) return p.data.tip;
      if (s && s.zfTips && s.zfTips[p.dataIndex] != null) return s.zfTips[p.dataIndex];
      return p.seriesName || "";
    };
    delete option.zf;
    return option;
  }
  var measure = document.createElement("canvas").getContext("2d");
  function textWidth(text, font) { measure.font = font; return measure.measureText(String(text)).width; }
  //: A wrapped legend's rows, apart.
  var LEGEND_ROW = 22;
  function legendRows(legend, width) {
    // ECharts lays a plain legend out in rows as wide as the chart: count them ahead of time.
    var rows = 1, used = 0, room = width - 16;
    legend.data.forEach(function (name) {
      var item = legend.itemWidth + 5 + textWidth(name, "11px Arial");
      if (used && used + 8 + item > room) { rows += 1; used = item; } else used += (used ? 8 : 0) + item;
    });
    return rows;
  }
  function plan(source, width, base) {
    // The chart laid out for the width it has: on a phone the legend wraps instead of paging, the title
    // shrinks to fit, a wide gutter of names is cut down, crowded labels turn, the toolbox goes.
    var option = hydrate(JSON.parse(source)), narrow = width < 520, extra = 0;
    var size = 14, rows = 1, gutter = 0, turned = 0, grid = option.grid;
    if (option.title && option.title.text) {
      // Centred, clear of the page's two buttons in the corner.
      while (size > 11 && textWidth(option.title.text, "bold " + size + "px Arial") > width - 124) size -= 1;
      option.title.textStyle.fontSize = size;
    }
    if (option.toolbox) option.toolbox.show = !narrow;
    if (option.legend) {
      rows = legendRows(option.legend, width);
      option.legend.type = rows > 4 ? "scroll" : "plain";
      option.legend.itemGap = 8;
      if (rows > 1 && rows <= 4) { extra += (rows - 1) * LEGEND_ROW; if (grid) grid.top += (rows - 1) * LEGEND_ROW; }
    }
    if (grid && option.yAxis && narrow && grid.left > width * 0.34) {
      var cap = Math.round(width * 0.34), cut = grid.left - cap;
      grid.left = cap;
      option.yAxis.axisLabel.width = cap - (option.yAxis.name ? 26 : 6) - 8;
      option.yAxis.axisLabel.overflow = "truncate";
      if (option.yAxis.nameGap) option.yAxis.nameGap = Math.max(4, option.yAxis.nameGap - cut);
      gutter = cap;
    }
    if (grid && option.xAxis) {
      var plot = width - grid.left - grid.right, labels = option.xAxis.axisLabel.zfLabels;
      if (option.xAxis.name) option.xAxis.nameTruncate = {maxWidth: plot, ellipsis: "…"};
      if (labels && labels.length > 2) {
        var widest = Math.max.apply(null, labels.map(function (l) { return textWidth(l[1].split("\n")[0], "11px Arial"); }));
        if (widest + 8 > plot / labels.length) {
          turned = Math.max(0, Math.round(widest * 0.72) - 6);
          option.xAxis.axisLabel.rotate = 45;
          grid.bottom += turned;
          if (option.xAxis.nameGap) option.xAxis.nameGap += turned;
          extra += turned;
        }
      }
    }
    var first = option.series && option.series[0];
    if (first && first.type === "treemap" && narrow) {
      extra += Math.max(0, Math.round(width * 1.15) - base);
      first.label.fontSize = 11;
      first.label.lineHeight = 13;
    }
    return {option: option, height: base + extra, key: [size, narrow, rows, gutter, turned].join("|")};
  }
  function draw(box) {
    if (box.zfDrawn || !box.offsetWidth) return box.zfChart;
    var source = box.querySelector('script[type="application/json"]');
    if (!source) return null;
    box.zfSource = source.textContent;
    box.zfHeight = box.offsetHeight;
    box.zfDrawn = true;
    var planned = plan(box.zfSource, box.offsetWidth, box.zfHeight);
    box.style.height = planned.height + "px";
    var chart = echarts.init(box, null, {renderer: "svg"});
    chart.setOption(planned.option);
    box.zfChart = chart;
    box.zfKey = planned.key;
    charts.push(box);
    return chart;
  }
  function refit(box) {
    var planned = plan(box.zfSource, box.offsetWidth, box.zfHeight);
    if (planned.key === box.zfKey) { box.zfChart.resize(); return; }
    box.style.height = planned.height + "px";
    box.zfChart.resize();
    box.zfChart.setOption(planned.option, true);
    box.zfKey = planned.key;
  }
  var boxes = Array.prototype.slice.call(document.querySelectorAll(".ichart"));
  if ("IntersectionObserver" in window) {
    var seen = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { draw(e.target); seen.unobserve(e.target); } });
    }, {rootMargin: "300px"});
    boxes.forEach(function (b) { seen.observe(b); });
  }
  document.querySelectorAll("details").forEach(function (d) {
    d.addEventListener("toggle", function () {
      if (d.open) d.querySelectorAll(".ichart").forEach(function (b) { setTimeout(function () { draw(b); }, 0); });
    });
  });
  var resizing;
  window.addEventListener("resize", function () {
    clearTimeout(resizing);
    resizing = setTimeout(function () { charts.forEach(function (box) { if (box.offsetWidth) refit(box); }); }, 120);
  });
  window.addEventListener("beforeprint", function () { boxes.forEach(draw); charts.forEach(function (box) { box.zfChart.resize(); }); });
  window.zipfolioDraw = draw;
  // The chart again, larger, in ``target`` (the page's enlarged view): laid out for its width, as tall as
  // the window allows, with zoom by wheel or pinch and panning by drag on both axes, and a reset.
  window.zipfolioEnlarge = function (box, target) {
    if (!box.zfSource) draw(box);
    if (!box.zfSource) return null;
    var width = target.clientWidth;
    // Wider than the page's column, taller in proportion; on a phone, most of the screen's height.
    var scale = Math.min(Math.max(width / 620, 1), 2);
    var base = width < 520 ? window.innerHeight - 170 : box.zfHeight * scale;
    base = Math.round(Math.min(window.innerHeight - 150, base));
    var planned = plan(box.zfSource, width, Math.max(base, 260)), option = planned.option;
    if (option.xAxis && option.yAxis) {
      var zoom = [{type: "inside", xAxisIndex: 0, filterMode: "none"}];
      if (option.xAxis.type !== "time") zoom.push({type: "inside", yAxisIndex: 0, filterMode: "none"});
      option.dataZoom = (option.dataZoom || []).concat(zoom);
      option.toolbox = {show: true, right: 4, top: 2, itemSize: 13, iconStyle: {borderColor: "#999999"},
                        feature: {restore: {title: "Reset zoom"}}};
    }
    target.style.height = Math.min(planned.height, window.innerHeight - 110) + "px";
    var chart = echarts.init(target, null, {renderer: "svg"});
    chart.setOption(option);
    return chart;
  };
  window.zipfolioDrawAll = function () { boxes.forEach(draw); return charts.length; };
})();
"""
