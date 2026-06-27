"""
optim_results.py

Tab de optimización GA. Contiene:
  - RoomOptimizationTab: contenedor con el stack (config GA <-> resultados)
  - ResultsScreen y sus paneles (Floor Plan 2D, Vista 3D, Frequency Response,
    Info panel, Opciones), antes en results_screen.py, ahora fusionados aquí.
  - OptimResultsWindow: ventana standalone equivalente a ResultsScreen, para
    abrir resultados guardados sin pasar por una corrida de GA.

Firma de dumF.run_ga_optimization:
    run_ga_optimization(geom_data, ga_config, min_mic_distance)
        geom_data:        dict   # geometría de la sala (vertices, walls, Z,
                                  #   audience_area, source_pos, etc.)
        ga_config:        dict   # vertex_ranges, audience_ranges, wall_ranges,
                                  #   height_ranges, output_dir, min_mic_distance
        min_mic_distance: float  # distancia mínima [m] entre micrófonos (3er
                                  #   argumento posicional, además de estar en
                                  #   ga_config["min_mic_distance"])

    Retorno: list[dict] — uno o más recintos candidatos, cada uno con:
        room_name:          str
        best_mic_positions:  list[list[float]]   # [[x, y, z], ...]
        params:              dict                # geometría optimizada del
                                                  #   recinto (mismo formato que
                                                  #   room_geometry)
        best_msfd:           float

Firma de dumF.get_resposes(room_params, room_name):
    room_params: dict con claves "best_mic_positions", "params", "best_msfd"
                 (mismo formato que se guarda/carga en los .json de resultado)
    room_name:   str

    Retorno: responses: list[dict] — uno por micrófono, cada dict:
        N_mic:    str    -> "Mic 1", "Mic 2", ...
        position: tuple  -> (x, y, z), numérica. Coincide en orden con
                             bestMicPositions (índice i -> mic i+1), que
                             sigue siendo la fuente de verdad para los demás
                             paneles (planta, 3D), pero ya no es necesario
                             ignorar este campo.
        freqs:    array  -> frecuencias [Hz]
        spl:      array  -> nivel de presión sonora [dB]

    IMPORTANTE: get_resposes() YA NO devuelve (bestMsfd, params, bestMicPositions,
    responses) como tupla de 4 elementos. bestMsfd, params y bestMicPositions
    deben tomarse directamente de room_params (o del resultado de
    run_ga_optimization, según el caso); get_resposes() sólo aporta `responses`.

Fuentes (source_pos):
    geom_data["source_pos"] es siempre una lista de fuentes [[x, y, z], ...],
    una por cada fuente sonora (puede haber una o varias, todas almacenadas
    juntas en la misma lista). _extract_room_data() y _extract_audience_and_source()
    (en GA_config.py) la normalizan a source_positions: list[tuple[float, float, float]].
"""

import sys
import os
import re
import json
import math
import numpy as np
import matplotlib.pyplot as plt

from geometry import signed_area, compute_ceiling
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (necesario para projection="3d")
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QSizePolicy, QComboBox, QStackedWidget, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QToolButton, QMenu, QWidgetAction, QFileDialog, QMessageBox,
    QApplication
)

# Añadimos el directorio actual al path e importamos tus funciones dummy
sys.path.insert(0, os.path.dirname(__file__))
import dummy_functions as dumF

# Importamos la configuración y componentes compartidos desde GA_config
from GA_config import GAConfigScreen, OptCanvas, _show_error


_MIC_COLOR_CYCLE = [
    "#ff6b6b", "#4dabf7", "#69db7c", "#ffd43b", "#da77f2",
    "#ff922b", "#3bc9db", "#f783ac", "#94d82d", "#748ffc",
]


def _mic_color(i: int) -> str:
    return _MIC_COLOR_CYCLE[i % len(_MIC_COLOR_CYCLE)]


def _export_canvas_to_file(parent: QWidget, canvas: FigureCanvas, default_name: str,
                            default_dir: str = "", has_data: bool = True):
    """Exporta el contenido de un FigureCanvas (matplotlib) a PNG/PDF/SVG vía
    diálogo de guardado. Lógica compartida entre ResultsScreen y
    OptimResultsWindow para no duplicar el flujo de exportación."""
    if not has_data:
        _show_error(parent, "Load a room first before exporting charts.")
        return

    path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save chart",
        os.path.join(default_dir, f"{default_name}.png"),
        "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)",
    )
    if not path:
        return

    try:
        canvas.figure.savefig(
            path,
            dpi=200,
            bbox_inches="tight",
            facecolor=canvas.figure.get_facecolor(),
        )
    except Exception as e:
        _show_error(parent, f"Could not save chart:\n{e}")
        return

    QMessageBox.information(parent, "Chart saved", f"Saved to:\n{path}")


def _mic_number(n_mic) -> int:
    """Extrae el número entero de N_mic, sin importar si llega como int,
    numpy scalar, o string tipo 'Mic 1'."""
    if isinstance(n_mic, str):
        match = re.search(r"\d+", n_mic)
        if match:
            return int(match.group())
        raise ValueError(f"Could not extract a number from N_mic={n_mic!r}")
    return int(n_mic)


def _extract_room_data(room_geometry: dict):
    """Extrae (geom, verts, height, tilts, audience_verts, source_positions) de
    room_geometry. Los tilts se indexan explícitamente por W{i+1} para no
    depender del orden de inserción del dict. audience_area usa las mismas
    claves V1, V2... que vertices (no A1 — esas son solo etiquetas de dibujo).
    source_positions es siempre una lista de tuplas (x, y, z): puede haber una
    o varias fuentes, todas almacenadas juntas en geom["source_pos"]."""
    geom   = room_geometry.get("data", room_geometry) if room_geometry else {}
    verts  = [(v[0], v[1]) for v in geom.get("vertices", {}).values()]
    height = geom.get("Z", 3.0)
    walls  = geom.get("walls", {})
    tilts  = [walls.get(f"W{i+1}", 0.0) for i in range(len(verts))]
    audience_verts = [(v[0], v[1]) for v in geom.get("audience_area", {}).values()]
    source_pos = geom.get("source_pos")
    if source_pos:
        first = source_pos[0]
        if isinstance(first, (list, tuple)):
            source_positions = [tuple(s) for s in source_pos]
        else:
            source_positions = [tuple(source_pos)]
    else:
        source_positions = []
    return geom, verts, height, tilts, audience_verts, source_positions


# ── Planta 2D ──────────────────────────────────────────────────────────────────

class PlanCanvas(FigureCanvas):
    """Vista en planta (X,Y) de la sala + posiciones de micrófonos.
    Mismo estilo visual que OptCanvas (GA_config.py): relleno del polígono,
    contorno y vértices numerados; los mics se dibujan como capa adicional."""

    def __init__(self, parent=None):
        fig = Figure(facecolor="#1e1e1e")
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = fig.add_subplot(111)
        self._style_ax()
        self.verts: list = []
        self.mics:  list = []   # [{"n": int, "pos": (x,y,z), "visible": bool}, ...]
        self.audience_verts: list = []
        self.source_positions: list = []

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="#333333")

    def load(self, verts, mics, audience_verts=None, source_positions=None):
        self.verts = list(verts)
        self.mics  = list(mics)
        self.audience_verts = list(audience_verts) if audience_verts is not None else []
        self.source_positions = [tuple(s) for s in source_positions] if source_positions else []
        self.redraw()

    def set_visibility(self, n_mic, visible: bool):
        for m in self.mics:
            if m["n"] == n_mic:
                m["visible"] = visible
        self.redraw()

    def redraw(self):
        ax = self.ax
        ax.cla()
        self._style_ax()

        verts = self.verts
        n     = len(verts)
        if n == 0:
            self.draw_idle()
            return
        xs, ys = [v[0] for v in verts], [v[1] for v in verts]

        if n >= 3:
            from matplotlib.patches import Polygon as P
            ax.add_patch(P(list(zip(xs, ys)), closed=True,
                           facecolor="#2a3f54", edgecolor="#aaaaaa",
                           linewidth=1.5, alpha=0.6, zorder=1))
        for i in range(n):
            j      = (i + 1) % n
            x1, y1 = verts[i]
            x2, y2 = verts[j]
            ax.plot([x1, x2], [y1, y2], color="#aaaaaa", linewidth=1.5, zorder=3)
        for i, (x, y) in enumerate(verts):
            ax.scatter([x], [y], color="#ffffff", s=30, zorder=5)
            ax.text(x, y, f" V{i+1}", color="#cccccc", fontsize=8, zorder=6)

        all_x, all_y = list(xs), list(ys)

        # Audience area (mismo estilo que room_design.py / GA_config.py)
        av = self.audience_verts
        na = len(av)
        if na >= 3:
            ax_, ay_ = [v[0] for v in av], [v[1] for v in av]
            from matplotlib.patches import Polygon as P
            ax.add_patch(P(list(zip(ax_, ay_)), closed=True,
                           facecolor="#5a8f3c", edgecolor="#8fd14f",
                           linewidth=1.5, alpha=0.35, zorder=2))
        for i in range(na):
            j      = (i + 1) % na
            x1, y1 = av[i]
            x2, y2 = av[j]
            ax.plot([x1, x2], [y1, y2], color="#8fd14f", linewidth=1.3, zorder=3)
        for i, (x, y) in enumerate(av):
            ax.scatter([x], [y], color="#8fd14f", s=40, marker="s", zorder=4)
            ax.text(x, y, f" A{i+1}", color="#8fd14f", fontsize=7, zorder=4)
            all_x.append(x); all_y.append(y)

        # Source(s)
        multi_src = len(self.source_positions) > 1
        for i, sp in enumerate(self.source_positions):
            sx, sy = sp[0], sp[1]
            ax.scatter([sx], [sy], color="#ff9d00", s=90, marker="*",
                       edgecolor="#1e1e1e", linewidth=0.8, zorder=6)
            label = f" Source {i+1}" if multi_src else " Source"
            ax.text(sx, sy, label, color="#ff9d00", fontsize=8,
                    fontweight="bold", zorder=6)
            all_x.append(sx); all_y.append(sy)

        for m in self.mics:
            if not m.get("visible", True):
                continue
            x, y, _z = m["pos"]
            color = _mic_color(m["n"])
            ax.scatter([x], [y], color=color, s=70, marker="^", zorder=7,
                       edgecolors="#1e1e1e", linewidths=0.8)
            ax.text(x, y, f" M{m['n']}", color=color, fontsize=8, zorder=8)
            all_x.append(x); all_y.append(y)

        if all_x and all_y:
            pad = 1.0
            ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        self.draw_idle()


# ── Vista 3D ───────────────────────────────────────────────────────────────────

class Room3DPanel(FigureCanvas):
    """Vista 3D de la sala + posiciones de micrófonos, embebida con matplotlib
    (mpl_toolkits.mplot3d). Sin dependencias externas, sin ventanas aparte."""

    def __init__(self, parent=None):
        fig = Figure(facecolor="#1e1e1e")
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = fig.add_subplot(111, projection="3d")
        self._style_ax()
        self.verts:  list = []
        self.room_height: float = 3.0
        self.tilts:  list = []
        self.mics:   list = []   # [{"n": int, "pos": (x,y,z), "visible": bool}, ...]
        self.audience_verts: list = []
        self.source_positions: list = []

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor("#1e1e1e")
        ax.xaxis.set_pane_color((0.12, 0.12, 0.12, 1.0))
        ax.yaxis.set_pane_color((0.12, 0.12, 0.12, 1.0))
        ax.zaxis.set_pane_color((0.12, 0.12, 0.12, 1.0))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_color("#aaaaaa")
            axis._axinfo["grid"]["color"] = (0.27, 0.27, 0.27, 1.0)
        ax.tick_params(colors="#aaaaaa")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

    def load(self, verts, height, tilts, mics, audience_verts=None, source_positions=None):
        self.verts  = list(verts)
        self.room_height = height
        self.tilts  = list(tilts)
        self.mics   = list(mics)
        self.audience_verts = list(audience_verts) if audience_verts is not None else []
        self.source_positions = [tuple(s) for s in source_positions] if source_positions else []
        self.redraw()

    def set_visibility(self, n_mic, visible: bool):
        for m in self.mics:
            if m["n"] == n_mic:
                m["visible"] = visible
        self.redraw()

    def redraw(self):
        ax = self.ax
        ax.cla()
        self._style_ax()

        if len(self.verts) >= 3:
            floor, ceiling = compute_ceiling(self.verts, self.room_height, self.tilts)
            n = len(floor)
            floor_pts   = [(x, y, 0.0) for x, y in floor]
            ceiling_pts = list(ceiling)

            faces = [floor_pts, ceiling_pts]
            for i in range(n):
                j = (i + 1) % n
                faces.append([floor_pts[i], floor_pts[j], ceiling_pts[j], ceiling_pts[i]])

            poly = Poly3DCollection(
                faces, facecolor="#9e9e9e", edgecolor="#555555",
                linewidths=0.6, alpha=0.25
            )
            ax.add_collection3d(poly)

            all_x = [p[0] for p in floor_pts]
            all_y = [p[1] for p in floor_pts]
            all_z = [0.0, self.room_height]
        else:
            all_x, all_y, all_z = [0.0], [0.0], [0.0, self.room_height]

        # Audience area: polígono plano en Z = 0 (no tiene componente Z)
        av = self.audience_verts
        if len(av) >= 3:
            aud_face = [(x, y, 0.0) for x, y in av]
            aud_poly = Poly3DCollection(
                [aud_face], facecolor="#5a8f3c", edgecolor="#8fd14f",
                linewidths=1.0, alpha=0.45
            )
            ax.add_collection3d(aud_poly)
            all_x += [p[0] for p in aud_face]
            all_y += [p[1] for p in aud_face]

        # Source(s)
        multi_src = len(self.source_positions) > 1
        for i, sp in enumerate(self.source_positions):
            sx, sy, sz = sp
            ax.scatter([sx], [sy], [sz], color="#ff9d00", s=80, marker="*",
                       depthshade=False, zorder=6)
            label = f" Source {i+1}" if multi_src else " Source"
            ax.text(sx, sy, sz, label, color="#ff9d00", fontsize=8, fontweight="bold")
            all_x.append(sx); all_y.append(sy); all_z.append(sz)

        for m in self.mics:
            if not m.get("visible", True):
                continue
            x, y, z = m["pos"]
            color = _mic_color(m["n"])
            ax.scatter([x], [y], [z], color=color, s=50, depthshade=False, zorder=5)
            ax.text(x, y, z, f" M{m['n']}", color=color, fontsize=8)
            all_x.append(x); all_y.append(y); all_z.append(z)

        if all_x and all_y and all_z:
            cx = (min(all_x) + max(all_x)) / 2
            cy = (min(all_y) + max(all_y)) / 2
            span = max(max(all_x) - min(all_x), max(all_y) - min(all_y),
                       max(all_z) - min(all_z), 1.0) / 2 * 1.2
            ax.set_xlim(cx - span, cx + span)
            ax.set_ylim(cy - span, cy + span)
            ax.set_zlim(min(0.0, min(all_z)), max(all_z) + 0.3)

        self.draw_idle()


# ── Respuesta en frecuencia ────────────────────────────────────────────────────

class FreqResponseCanvas(FigureCanvas):
    """SPL [dB] vs frecuencia [Hz], una curva por mic visible.

    Eje X logarítmico con ticks "humanos" (sin notación científica): frecuencias
    centrales normalizadas de 1/3 de octava si el rango de frecuencias es
    amplio (>= ~2 décadas, típico de respuesta acústica completa), o múltiplos
    de 100 Hz si el rango es más acotado. Al pasar el mouse sobre una curva se
    muestra un tooltip con el valor de SPL de cada mic visible en esa frecuencia."""

    # Frecuencias centrales normalizadas de 1/3 de octava (ISO 266 / IEC 61260),
    # usadas como ticks cuando el rango de frecuencias de los datos es amplio.
    _OCTAVE_TICKS = [
        16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
        500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
        8000, 10000, 12500, 16000, 20000,
    ]

    def __init__(self, parent=None):
        fig = Figure(facecolor="#1e1e1e")
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = fig.add_subplot(111)
        self._style_ax()
        self.responses: list = []

        # ── Tooltip interactivo (hover) ─────────────────────────────────
        self._cursor_lines: list = []   # líneas matplotlib trazadas (para hit-test)
        self._annotation = None
        self._build_annotation()
        self.mpl_connect("motion_notify_event", self._on_hover)
        self.mpl_connect("axes_leave_event", self._on_leave)

    def _build_annotation(self):
        self._annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(14, 14), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#2b2b2b", ec="#555555", alpha=0.95),
            color="#eeeeee", fontsize=8, zorder=10,
        )
        self._annotation.set_visible(False)

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("SPL [dB]")
        ax.grid(True, color="#333333", which="both")

    def load(self, responses):
        self.responses = list(responses)
        self.redraw()

    # ── Elección de ticks "humanos" para el eje X ───────────────────────────
    def _x_ticks_for_range(self, fmin: float, fmax: float):
        """Devuelve la lista de ticks a usar en el eje X según el rango de
        frecuencias presente en los datos: frecuencias centrales de 1/3 de
        octava normalizadas si el rango es amplio (típico en acústica), o
        múltiplos de 100 Hz si es acotado."""
        if fmax / max(fmin, 1e-9) >= 8:
            # Rango amplio (más de ~3 octavas): usar tercios de octava normalizados
            ticks = [t for t in self._OCTAVE_TICKS if fmin * 0.9 <= t <= fmax * 1.1]
            if len(ticks) >= 2:
                return ticks
        # Rango acotado: ticks cada 100 Hz (o cada 50 Hz si el rango es chico)
        step = 100.0 if (fmax - fmin) > 300 else 50.0
        start = math.floor(fmin / step) * step
        end   = math.ceil(fmax / step) * step
        ticks = list(np.arange(max(start, step), end + step, step))
        return ticks

    def redraw(self):
        ax = self.ax
        ax.cla()
        self._style_ax()
        ax.set_xscale("log")

        all_freqs = []
        any_curve = False
        for r in self.responses:
            if not r.get("visible", True):
                continue
            freqs = r.get("freqs")
            spl   = r.get("spl")
            if freqs is None or spl is None or len(freqs) == 0 or len(spl) == 0:
                continue
            any_curve = True
            all_freqs.extend(freqs)
            ax.plot(freqs, spl, color=_mic_color(r["N_mic"]),
                    linewidth=1.4, label=r.get("label", f"Mic {r['N_mic']}"))

        if any_curve:
            ax.legend(facecolor="#2b2b2b", edgecolor="#444444",
                      labelcolor="#dddddd", fontsize=8)

            fmin, fmax = min(all_freqs), max(all_freqs)
            ticks = self._x_ticks_for_range(fmin, fmax)
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticks([], minor=True)  # sin sub-ticks log automáticos

            def _fmt_hz(val, _pos=None):
                # Sin notación científica: enteros simples, o con 1 decimal
                # si hace falta (p.ej. 31.5 Hz de la serie de tercios de octava).
                return f"{val:.0f}" if float(val).is_integer() else f"{val:g}"

            ax.xaxis.set_major_formatter(FuncFormatter(_fmt_hz))
            # Con la serie de tercios de octava hay más ticks que antes:
            # reducimos la fuente para que entren sin superponerse.
            ax.tick_params(axis="x", labelsize=7)

        # Reconstruir la anotación del tooltip (se perdió con ax.cla())
        self._build_annotation()
        self.draw_idle()

    # ── Tooltip interactivo ──────────────────────────────────────────────
    def _on_leave(self, _event):
        if self._annotation is not None and self._annotation.get_visible():
            self._annotation.set_visible(False)
            self.draw_idle()

    def _on_hover(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            self._on_leave(event)
            return

        visible_responses = [
            r for r in self.responses
            if r.get("visible", True) and r.get("freqs") is not None
            and r.get("spl") is not None and len(r.get("freqs")) > 0
        ]
        if not visible_responses:
            return

        x_cursor = event.xdata
        lines = []
        for r in visible_responses:
            freqs = np.asarray(r["freqs"], dtype=float)
            spl   = np.asarray(r["spl"], dtype=float)
            # Búsqueda en escala logarítmica (eje X es log)
            idx = int(np.argmin(np.abs(np.log10(np.maximum(freqs, 1e-9)) - np.log10(max(x_cursor, 1e-9)))))
            f_val, s_val = freqs[idx], spl[idx]
            label = r.get("label", f"Mic {r['N_mic']}")
            color = _mic_color(r["N_mic"])
            lines.append((f_val, label, s_val, color))

        if not lines:
            return

        # Frecuencia de referencia: la del mic más cercano al cursor
        f_ref = min(lines, key=lambda t: abs(t[0] - x_cursor))[0]
        text_lines = [f"f ≈ {f_ref:.0f} Hz"]
        for f_val, label, s_val, _color in lines:
            text_lines.append(f"{label}: {s_val:.1f} dB")

        self._annotation.xy = (f_ref, lines[0][2])
        self._annotation.set_text("\n".join(text_lines))
        self._annotation.set_visible(True)
        self.draw_idle()


# ── Panel de info ──────────────────────────────────────────────────────────────

class InfoPanel(QWidget):
    """Room params (geometría óptima devuelta por el GA), MSFD, y tabla de
    mejores posiciones de mic. El botón para cargar un recinto vive en
    OptionsPanel, que llama a InfoPanel.load_room().

    IMPORTANTE sobre las señales: usamos Signal(object) en vez de
    Signal(float, dict, list, list). PySide6 intenta convertir los tipos
    declarados en una Signal a sus equivalentes C++ (QVariantMap, QVariantList,
    etc.), y esa conversión falla con el warning
        "_pythonToCppCopy: Cannot copy-convert ... (dict) to C++"
    en cuanto el dict/list contiene algo que no sea un tipo primitivo trivial
    (numpy arrays, tuplas, dicts anidados con valores no estándar, etc. — como
    los arrays de freqs/spl que vienen en `responses`). Declarando la señal
    como `object` evitamos por completo ese marshalling: Qt pasa la referencia
    de Python tal cual, sin intentar reconstruirla como tipo C++.
    """

    # Se emite tras cargar un .json y llamar a get_resposes() exitosamente,
    # para que el contenedor (ResultsScreen / OptimResultsWindow) refresque
    # el resto de los paneles (planta, 3D, frecuencia, opciones).
    # Payload: tupla (bestMsfd: float, params: dict, bestMicPositions: list, responses: list)
    roomLoaded = Signal(object)

    # Se emiten alrededor de la carga de un .json (selección de archivo +
    # lectura + get_resposes), para que el contenedor pueda mostrar/ocultar
    # un overlay de "Cargando…" mientras dura la operación.
    loadStarted  = Signal()
    loadFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params: dict = None  # último `params` recibido (formato room_geometry)
        self._default_dir: str = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        params_box = QGroupBox("Room params")
        pl = QVBoxLayout(params_box)
        self.params_lbl = QLabel("—")
        self.params_lbl.setStyleSheet("color: #cccccc;")
        self.params_lbl.setWordWrap(True)
        pl.addWidget(self.params_lbl)

        lay.addWidget(params_box)

        msfd_box = QGroupBox("MSFD")
        ml = QVBoxLayout(msfd_box)
        self.msfd_lbl = QLabel("—")
        self.msfd_lbl.setStyleSheet("color: #69db7c; font-size: 16pt; font-weight: bold;")
        self.msfd_lbl.setAlignment(Qt.AlignCenter)
        ml.addWidget(self.msfd_lbl)
        lay.addWidget(msfd_box)

        mics_box = QGroupBox("Best mics positions")
        mil = QVBoxLayout(mics_box)
        self.mics_table = QTableWidget(0, 4)
        self.mics_table.setHorizontalHeaderLabels(["Mic", "X", "Y", "Z"])
        self.mics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.mics_table.verticalHeader().setVisible(False)
        self.mics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        mil.addWidget(self.mics_table)
        lay.addWidget(mics_box, stretch=1)

    def set_default_dir(self, path: str):
        self._default_dir = path or ""

    def load(self, bestMsfd, mics, params):
        """params: dict con el mismo formato que room_geometry (incluye la
        clave "data" con vertices, walls, audience_area, Z, source_pos)."""
        self._params = params
        geom = (params or {}).get("data", params) if params else {}

        n_verts = len(geom.get("vertices", {})) if geom else 0
        height  = geom.get("Z", "—") if geom else "—"
        n_walls = len(geom.get("walls", {})) if geom else 0
        n_aud   = len(geom.get("audience_area", {})) if geom else 0
        src     = geom.get("source_pos", []) if geom else []
        n_src   = len(src)

        lines = [
            f"Vertices: {n_verts}",
            f"Walls: {n_walls}",
            f"Height (Z): {height} m",
            f"Audience verts: {n_aud}",
            f"Sources: {n_src}",
        ]
        for i, sp in enumerate(src):
            x, y, z = sp
            lines.append(f"  Source {i+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
        self.params_lbl.setText("\n".join(lines))

        self.msfd_lbl.setText(f"{bestMsfd:.4f}" if isinstance(bestMsfd, (int, float)) else "—")

        self.mics_table.setRowCount(0)
        for m in mics:
            row = self.mics_table.rowCount()
            self.mics_table.insertRow(row)
            x, y, z = m["pos"]
            for c, val in enumerate([f"M{m['n']}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"]):
                self.mics_table.setItem(row, c, QTableWidgetItem(val))

    # ── Cargar recinto desde .json (misma lógica que tenía OptimResultsWindow) ─
    # Público: lo invoca el botón "Load Room…" alojado en OptionsPanel.
    def load_room(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load room result", self._default_dir, "JSON Files (*.json)"
        )
        if not path:
            return

        self.loadStarted.emit()
        try:
            room_name = os.path.splitext(os.path.basename(path))[0]

            try:
                with open(path, encoding="utf-8") as f:
                    room_data = json.load(f)
            except Exception as e:
                _show_error(self, f"Could not read file:\n{e}")
                return

            room_params = room_data  # dict con best_mic_positions, params, best_msfd
            bestMsfd         = room_params.get("best_msfd")
            params            = room_params.get("params")
            bestMicPositions  = room_params.get("best_mic_positions", [])

            try:
                responses = dumF.get_resposes(room_params, room_name)
            except Exception as e:
                _show_error(self, f"Error calling get_resposes:\n{e}")
                return

            self.roomLoaded.emit((bestMsfd, params, bestMicPositions, responses))
        finally:
            self.loadFinished.emit()


# ── Panel de opciones ──────────────────────────────────────────────────────────

class OptionsPanel(QWidget):
    """Dropdown con checkboxes para elegir qué mic(s) se muestran en planta,
    3D y respuesta en frecuencia, botones de Load Room / Export Frequency
    Response, más el botón para volver a la config GA."""

    backRequested = Signal()

    def __init__(self, on_toggle, on_load_room=None, on_export_freq=None, parent=None):
        super().__init__(parent)
        self._on_toggle = on_toggle
        self._checks: dict = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Options")
        bl  = QVBoxLayout(box)

        # ── Dropdown "Show mics" ──────────────────────────────────────────
        self.mics_menu = QMenu(self)
        self.mics_btn = QToolButton()
        self.mics_btn.setText("Show mics ▾")
        self.mics_btn.setPopupMode(QToolButton.InstantPopup)
        self.mics_btn.setMenu(self.mics_menu)
        self.mics_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bl.addWidget(self.mics_btn)

        btn_row = QHBoxLayout()
        self.btn_all  = QPushButton("All")
        self.btn_none = QPushButton("None")
        self.btn_all.clicked.connect(lambda: self._set_all(True))
        self.btn_none.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        bl.addLayout(btn_row)

        # ── Load Room / Export Frequency Response ──────────────────────────
        self.btn_load_room = QPushButton("Load Room…")
        self.btn_load_room.setProperty("role", "secondary")
        if on_load_room is not None:
            self.btn_load_room.clicked.connect(on_load_room)
        bl.addWidget(self.btn_load_room)

        self.btn_export_freq = QPushButton("Export Frequency Response…")
        self.btn_export_freq.setProperty("role", "secondary")
        if on_export_freq is not None:
            self.btn_export_freq.clicked.connect(on_export_freq)
        bl.addWidget(self.btn_export_freq)

        bl.addStretch(1)

        # ── Volver a la config del GA ─────────────────────────────────────
        self.btn_back = QPushButton("← Back to GA config")
        self.btn_back.setProperty("role", "secondary")
        self.btn_back.clicked.connect(self.backRequested.emit)
        bl.addWidget(self.btn_back)

        root.addWidget(box, stretch=1)

    def rebuild(self, mic_numbers: list):
        self.mics_menu.clear()
        self._checks.clear()

        for n in mic_numbers:
            cb = QCheckBox(f"M{n}")
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {_mic_color(n)};")
            cb.toggled.connect(lambda checked, n=n: self._on_toggle(n, checked))
            action = QWidgetAction(self.mics_menu)
            action.setDefaultWidget(cb)
            self.mics_menu.addAction(action)
            self._checks[n] = cb

    def _set_all(self, checked: bool):
        for n, cb in self._checks.items():
            cb.setChecked(checked)  # dispara toggled -> _on_toggle


# ── Pantalla completa de resultados (embebida en una tab) ──────────────────────

class ResultsScreen(QWidget):
    """Pantalla de resultados embebida como tab ("Optimization Results" en
    main.py). Se puebla automáticamente cuando RoomOptimizationTab emite
    resultsReady, o manualmente con el botón "Cargar recinto…" del InfoPanel."""

    backToConfigRequested = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._mics: list = []        # [{"n", "pos", "visible"}, ...]
        self._responses: list = []   # copia local con "visible" agregado
        self._params: dict = None    # geometría optimizada (formato room_geometry)
        self._default_dir: str = ""

        # Overlay creado antes de _build para que exista cuando corre el layout
        # (mismo estilo/patrón que GAConfigScreen._overlay en GA_config.py).
        self._overlay = QLabel("Loading room…", self)
        self._overlay.setAlignment(Qt.AlignCenter)
        self._overlay.setStyleSheet("""
            background-color: rgba(0,0,0,200);
            color: white; font-size: 22pt; font-weight: bold; border-radius: 8px;
        """)
        self._overlay.hide()

        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Columna izquierda (gráficos) ──────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        plan_box = QGroupBox("Floor Plan")
        pl = QVBoxLayout(plan_box)
        self.plan_canvas = PlanCanvas()
        pl.addWidget(self.plan_canvas)
        top_row.addWidget(plan_box, stretch=1)

        view3d_box = QGroupBox("3D")
        v3l = QVBoxLayout(view3d_box)
        self.room3d = Room3DPanel()
        v3l.addWidget(self.room3d)
        top_row.addWidget(view3d_box, stretch=1)

        left.addLayout(top_row, stretch=3)

        freq_box = QGroupBox("Frequency Response")
        fql = QVBoxLayout(freq_box)
        self.freq_canvas = FreqResponseCanvas()
        fql.addWidget(self.freq_canvas)

        left.addWidget(freq_box, stretch=2)

        # Proporción 3:1 (igual que la columna canvas/control de GAConfigScreen)
        # para que los gráficos tengan más espacio.
        root.addLayout(left, stretch=3)

        # ── Columna derecha (info + opciones) ─────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(10)

        self.info_panel = InfoPanel()
        self.info_panel.roomLoaded.connect(self._on_room_loaded)
        self.info_panel.loadStarted.connect(self.show_overlay)
        self.info_panel.loadFinished.connect(self.hide_overlay)
        right.addWidget(self.info_panel, stretch=2)

        # "Load Room…" y "Export Frequency Response…" viven dentro de la
        # caja de Options (antes estaban en InfoPanel / debajo del gráfico).
        self.options_panel = OptionsPanel(
            self._on_mic_toggled,
            on_load_room=self.info_panel.load_room,
            on_export_freq=self._on_export_freq,
        )
        self.options_panel.backRequested.connect(self.backToConfigRequested.emit)
        right.addWidget(self.options_panel, stretch=1)

        root.addLayout(right, stretch=1)

    # ── Overlay de carga ───────────────────────────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.setGeometry(0, 0, self.width(), self.height())

    def show_overlay(self):
        self._overlay.setGeometry(0, 0, self.width(), self.height())
        self._overlay.show()
        self._overlay.raise_()
        QApplication.processEvents()

    def hide_overlay(self):
        self._overlay.hide()

    # ── Exportar gráfico de respuesta en frecuencia ────────────────────────
    def _on_export_freq(self):
        _export_canvas_to_file(self, self.freq_canvas, "freq_response",
                                self._default_dir, has_data=bool(self._params))

    # ── Slot intermedio: desempaqueta el payload de InfoPanel.roomLoaded ──────
    def _on_room_loaded(self, payload):
        bestMsfd, params, bestMicPositions, responses = payload
        self.load_results(bestMsfd, params, bestMicPositions, responses)

    # ── Carga de resultados ──────────────────────────────────────────────────
    def load_results(self, bestMsfd, params, bestMicPositions, responses):
        """params: dict con el mismo formato que room_geometry — es la
        geometría OPTIMIZADA del recinto devuelta por run_ga_optimization
        (no la geometría de entrada; reemplaza el uso de self.state.room_geometry
        para graficar y para mostrar/guardar el recinto resultante)."""
        self._params = params
        geom, verts, height, tilts, audience_verts, source_positions = _extract_room_data(params)

        # bestMicPositions es la fuente de verdad para las posiciones numéricas.
        # responses[i]["position"] también trae (x, y, z) numérico en el mismo
        # orden, pero se sigue usando bestMicPositions para no depender del
        # orden de responses.
        self._mics = [
            {"n": i + 1, "pos": tuple(pos), "visible": True}
            for i, pos in enumerate(bestMicPositions)
        ]

        self._responses = [
            dict(r, N_mic=_mic_number(r["N_mic"]), label=str(r["N_mic"]), visible=True)
            for r in responses
        ]

        self.plan_canvas.load(verts, self._mics, audience_verts, source_positions)
        self.room3d.load(verts, height, tilts, self._mics, audience_verts, source_positions)
        self.freq_canvas.load(self._responses)
        self.info_panel.load(bestMsfd, self._mics, params)

        mic_numbers = sorted({m["n"] for m in self._mics} | {r["N_mic"] for r in self._responses})
        self.options_panel.rebuild(mic_numbers)

    # ── Toggle de visibilidad de un mic ──────────────────────────────────────
    def _on_mic_toggled(self, n_mic, visible):
        for m in self._mics:
            if m["n"] == n_mic:
                m["visible"] = visible
        for r in self._responses:
            if r["N_mic"] == n_mic:
                r["visible"] = visible

        self.plan_canvas.set_visibility(n_mic, visible)
        self.freq_canvas.redraw()

        _geom, verts, height, tilts, audience_verts, source_positions = _extract_room_data(self._params)
        self.room3d.load(verts, height, tilts, self._mics, audience_verts, source_positions)


# ── Ventana standalone de resultados ────────────────────────────────────────────

class OptimResultsWindow(QWidget):
    """Ventana independiente que muestra los resultados de optimización.
    Se abre sin un recinto preseleccionado: el usuario levanta un .json con
    el botón 'Load room (.json)…'. Llama a get_resposes(room_params, room_name)
    para obtener los datos de respuesta a graficar.

    Hoy ya no se abre automáticamente al terminar una corrida de GA (eso lo
    maneja RoomOptimizationTab.resultsReady -> ResultsScreen, embebido en una
    tab); se deja disponible para abrir resultados guardados de forma
    independiente si hace falta.
    """

    def __init__(self, default_dir: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimization Results")
        self.resize(1280, 800)
        self._default_dir = default_dir
        self._mics: list = []
        self._responses: list = []
        self._params: dict = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Barra superior: cargar archivo ─────────────────────────────────
        top_bar = QHBoxLayout()
        self.file_lbl = QLabel("No room loaded — use the button to open a result file.")
        self.file_lbl.setStyleSheet("color: #888888; font-size: 9pt;")
        top_bar.addWidget(self.file_lbl, stretch=1)

        self.btn_load = QPushButton("Load room (.json)…")
        self.btn_load.setProperty("role", "secondary")
        self.btn_load.clicked.connect(self._load_room_file)
        top_bar.addWidget(self.btn_load)

        root.addLayout(top_bar)

        # ── Contenido principal ────────────────────────────────────────────
        content = QHBoxLayout()
        content.setSpacing(10)

        # Columna izquierda: gráficos
        left = QVBoxLayout()
        left.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        plan_box = QGroupBox("Floor Plan")
        pl = QVBoxLayout(plan_box)
        self.plan_canvas = PlanCanvas()
        pl.addWidget(self.plan_canvas)
        top_row.addWidget(plan_box, stretch=1)

        view3d_box = QGroupBox("3D")
        v3l = QVBoxLayout(view3d_box)
        self.room3d = Room3DPanel()
        v3l.addWidget(self.room3d)
        top_row.addWidget(view3d_box, stretch=1)

        left.addLayout(top_row, stretch=2)

        freq_box = QGroupBox("Frequency Response")
        fl = QVBoxLayout(freq_box)
        self.freq_canvas = FreqResponseCanvas()
        fl.addWidget(self.freq_canvas)
        left.addWidget(freq_box, stretch=1)

        content.addLayout(left, stretch=3)

        # Columna derecha: info + opciones + exportar
        right = QVBoxLayout()
        right.setSpacing(10)

        self.info_panel = InfoPanel()
        self.info_panel.set_default_dir(self._default_dir)
        self.info_panel.roomLoaded.connect(self._on_room_loaded)
        right.addWidget(self.info_panel, stretch=2)

        self.options_panel = OptionsPanel(
            self._on_mic_toggled,
            on_load_room=self.info_panel.load_room,
            on_export_freq=lambda: self._export_canvas(self.freq_canvas, "freq_response"),
        )
        # Esta ventana ya tiene su propio botón "Load room (.json)…" en la
        # barra superior; ocultamos el de OptionsPanel para no duplicarlo.
        self.options_panel.btn_load_room.hide()
        # Ya hay un botón "Save Freq…" abajo en "Export charts"; ocultamos
        # el de OptionsPanel para no duplicarlo.
        self.options_panel.btn_export_freq.hide()
        # En esta ventana no hay "back to config", ocultamos ese botón
        self.options_panel.btn_back.hide()
        right.addWidget(self.options_panel, stretch=1)

        # Panel de exportación
        export_box = QGroupBox("Export charts")
        el = QVBoxLayout(export_box)

        export_grid = QHBoxLayout()
        for label, cb in [
            ("Plan 2D", "_exp_plan"),
            ("3D",      "_exp_3d"),
            ("Freq",    "_exp_freq"),
        ]:
            btn = QPushButton(f"Save {label}…")
            btn.setProperty("role", "secondary")
            setattr(self, cb, btn)
            export_grid.addWidget(btn)

        self._exp_plan.clicked.connect(lambda: self._export_canvas(self.plan_canvas, "plan_2d"))
        self._exp_3d.clicked.connect(lambda: self._export_canvas(self.room3d, "room_3d"))
        self._exp_freq.clicked.connect(lambda: self._export_canvas(self.freq_canvas, "freq_response"))

        el.addLayout(export_grid)
        right.addWidget(export_box)

        content.addLayout(right, stretch=1)
        root.addLayout(content)

    # ── Cargar archivo .json (barra superior) ──────────────────────────────
    def _load_room_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load room result", self._default_dir, "JSON Files (*.json)"
        )
        if not path:
            return

        room_name = os.path.splitext(os.path.basename(path))[0]

        try:
            with open(path, encoding="utf-8") as f:
                room_data = json.load(f)
        except Exception as e:
            _show_error(self, f"Could not read file:\n{e}")
            return

        room_params = room_data  # dict con best_mic_positions, params, best_msfd
        bestMsfd         = room_params.get("best_msfd")
        params            = room_params.get("params")
        bestMicPositions  = room_params.get("best_mic_positions", [])

        try:
            responses = dumF.get_resposes(room_params, room_name)
        except Exception as e:
            _show_error(self, f"Error calling get_resposes:\n{e}")
            return

        self.file_lbl.setText(f"Loaded: {path}")
        self.file_lbl.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        self._load_results(bestMsfd, params, bestMicPositions, responses)

    # ── Slot intermedio: desempaqueta el payload de InfoPanel.roomLoaded ──────
    def _on_room_loaded(self, payload):
        bestMsfd, params, bestMicPositions, responses = payload
        self.file_lbl.setText("Loaded from InfoPanel")
        self.file_lbl.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        self._load_results(bestMsfd, params, bestMicPositions, responses)

    # ── Populate UI con los resultados ────────────────────────────────────
    def _load_results(self, bestMsfd, params, bestMicPositions, responses):
        self._params = params
        geom, verts, height, tilts, audience_verts, source_positions = _extract_room_data(params)

        self._mics = [
            {"n": i + 1, "pos": tuple(pos), "visible": True}
            for i, pos in enumerate(bestMicPositions)
        ]
        self._responses = [
            dict(r, N_mic=_mic_number(r["N_mic"]), label=str(r["N_mic"]), visible=True)
            for r in responses
        ]

        self.plan_canvas.load(verts, self._mics, audience_verts, source_positions)
        self.room3d.load(verts, height, tilts, self._mics, audience_verts, source_positions)
        self.freq_canvas.load(self._responses)
        self.info_panel.load(bestMsfd, self._mics, params)

        mic_numbers = sorted({m["n"] for m in self._mics} | {r["N_mic"] for r in self._responses})
        self.options_panel.rebuild(mic_numbers)

    # ── Toggle visibilidad de mic ─────────────────────────────────────────
    def _on_mic_toggled(self, n_mic, visible):
        for m in self._mics:
            if m["n"] == n_mic:
                m["visible"] = visible
        for r in self._responses:
            if r["N_mic"] == n_mic:
                r["visible"] = visible

        self.plan_canvas.set_visibility(n_mic, visible)
        self.freq_canvas.redraw()

        _geom, verts, height, tilts, audience_verts, source_positions = _extract_room_data(self._params)
        self.room3d.load(verts, height, tilts, self._mics, audience_verts, source_positions)

    # ── Exportar un canvas matplotlib como imagen ─────────────────────────
    def _export_canvas(self, canvas: FigureCanvas, default_name: str):
        _export_canvas_to_file(self, canvas, default_name, self._default_dir,
                                has_data=bool(self._params))


# ── Main tab (agrupa config GA y resultados) ──────────────────────────────────

class RoomOptimizationTab(QWidget):
    """Contenedor de la tab "Room Optimization": un QStackedWidget con la
    pantalla de configuración del GA (GAConfigScreen). Al terminar una corrida
    (runRequested), ejecuta el GA y emite resultsReady con los datos del mejor
    recinto encontrado, para que MainPage los muestre en la tab
    "Optimization Results" (ver main.py)."""

    # Payload: tupla (bestMsfd: float, params: dict, bestMicPositions: list, responses: list)
    # Ver nota sobre Signal(object) en InfoPanel — evita el marshalling
    # automático de Qt a tipos C++, que falla con estructuras Python anidadas.
    resultsReady = Signal(object)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.stack          = QStackedWidget()
        self.ga_screen      = GAConfigScreen(self.state)

        self.stack.addWidget(self.ga_screen)
        self.stack.setCurrentWidget(self.ga_screen)

        self.ga_screen.runRequested.connect(self._on_run_requested)
        root.addWidget(self.stack)

    def _on_back_to_config(self):
        self.stack.setCurrentWidget(self.ga_screen)

    def _on_run_requested(self):
        self.ga_screen.show_overlay()

        # Forzamos el sync por si quedó algún cambio sin disparar
        # (ej: Min Mic Distance tipeado o checkbox tocado sin abrir un diálogo de rango)
        self.ga_screen._sync_to_state()

        geom_data = dict(self.state.room_geometry)
        geom_data.setdefault("is_symmetric", self.state.symmetric)
        geom_data.setdefault("volume", 0.0)
        ga_config = self.state.ga_config  # vertex_ranges, wall_ranges, height_ranges, output_dir, min_mic_distance
        min_mic_distance = ga_config.get("min_mic_distance", 0.5)

        # cleaned_output: list[dict] — uno o más recintos candidatos.
        # Cada dict tiene las claves: "room_name", "best_mic_positions", "params", "best_msfd"
        cleaned_output = dumF.run_ga_optimization(
            geom_data, ga_config, min_mic_distance
        )

        # ── Guardar cada recinto como .json en el directorio de salida ─────────
        output_dir = ga_config.get("output_dir", "")
        if output_dir and os.path.isdir(output_dir):
            for room in cleaned_output:
                room_name = room.get("room_name", "room")
                file_path = os.path.join(output_dir, f"{room_name}.json")
                payload = {
                    "best_mic_positions": room.get("best_mic_positions"),
                    "params": room.get("params"),
                    "best_msfd": room.get("best_msfd"),
                }
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    _show_error(self, f"Could not save {room_name}.json:\n{e}")

        self.ga_screen.hide_overlay()

        # ── Mostrar resultados en la tab "Optimization Results" ────────────────
        # En vez de abrir una ventana aparte, emitimos los resultados del
        # mejor recinto (cleaned_output[0]) para que MainPage los muestre
        # en la tab "Optimization Results" y cambie automáticamente a ella.
        if not cleaned_output:
            _show_error(self, "Optimization did not return any room.")
            return

        best_room = cleaned_output[0]
        params             = best_room.get("params", {})
        best_mic_positions = best_room.get("best_mic_positions", [])
        best_msfd          = best_room.get("best_msfd", 0.0)

        try:
            responses = dumF.get_resposes(
                {
                    "best_mic_positions": best_mic_positions,
                    "params": params,
                    "best_msfd": best_msfd,
                },
                best_room.get("room_name", "room"),
            )
        except Exception as e:
            _show_error(self, f"Error calling get_resposes:\n{e}")
            return

        self.resultsReady.emit((best_msfd, params, best_mic_positions, responses))

    def _load_from_design(self):
        self.ga_screen._load_from_design()