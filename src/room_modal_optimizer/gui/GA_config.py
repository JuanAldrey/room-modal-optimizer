import json
import sys
import os
import math
import numpy as np

from room_modal_optimizer.gui.geometry import nearest_wall
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QFrame, QDialog, QFormLayout,
    QDialogButtonBox, QFileDialog, QMessageBox,
    QCheckBox, QAbstractItemView, QApplication
)

# ── Geometry extraction helper ──────────────────────────────────────────────────

def _extract_audience_and_source(geom: dict):
    """Extrae (audience_verts, source_positions) de geom. audience_area usa las
    mismas claves V1, V2... que vertices (no A1 — esas son solo etiquetas de
    dibujo). source_positions es siempre una lista de tuplas (x, y, z), una por
    cada fuente (puede haber una o varias, todas almacenadas juntas en
    geom["source_pos"])."""
    audience_verts = [(v[0], v[1]) for v in geom.get("audience_area", {}).values()]
    source_pos = geom.get("source_pos")
    if source_pos:
        # source_pos es una lista de fuentes [[x,y,z], ...]; soportamos también
        # el caso legacy de una sola fuente como [x,y,z] suelto.
        first = source_pos[0]
        if isinstance(first, (list, tuple)):
            source_positions = [tuple(s) for s in source_pos]
        else:
            source_positions = [tuple(source_pos)]
    else:
        source_positions = []
    return audience_verts, source_positions


# ── Error dialog ──────────────────────────────────────────────────────────────

def _show_error(parent, msg: str):
    dlg = QMessageBox(parent)
    dlg.setIcon(QMessageBox.Critical)
    dlg.setWindowTitle("Error")
    dlg.setText(msg)
    dlg.setStandardButtons(QMessageBox.Ok)
    dlg.button(QMessageBox.Ok).setText("OK")
    dlg.setStyleSheet("""
        QMessageBox { background-color: #2b2b2b; }
        QMessageBox QLabel { color: #ff6b6b; font-size: 10pt; }
        QPushButton {
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #5a5a5a, stop:1 #3d3d3d);
            color: #dddddd; border-radius: 4px;
            padding: 5px 15px; font-weight: bold; border: none;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                stop:0 #6a6a6a, stop:1 #4d4d4d);
        }
    """)
    dlg.exec()


# ── Range dialogs ─────────────────────────────────────────────────────────────

class VertexRangeDialog(QDialog):
    def __init__(self, idx, ranges, parent=None, label="V"):
        super().__init__(parent)
        self.setWindowTitle(f"{label}{idx+1} — Range")
        self.setMinimumWidth(260)
        dv   = QDoubleValidator(-999, 999, 3, self)
        form = QFormLayout(self)
        self.xmin = QLineEdit(str(ranges.get("xmin", 0.0))); self.xmin.setValidator(dv)
        self.xmax = QLineEdit(str(ranges.get("xmax", 0.0))); self.xmax.setValidator(dv)
        self.ymin = QLineEdit(str(ranges.get("ymin", 0.0))); self.ymin.setValidator(dv)
        self.ymax = QLineEdit(str(ranges.get("ymax", 0.0))); self.ymax.setValidator(dv)
        form.addRow("X min [m]:", self.xmin)
        form.addRow("X max [m]:", self.xmax)
        form.addRow("Y min [m]:", self.ymin)
        form.addRow("Y max [m]:", self.ymax)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_ranges(self):
        return {
            "xmin": float(self.xmin.text()), "xmax": float(self.xmax.text()),
            "ymin": float(self.ymin.text()), "ymax": float(self.ymax.text()),
        }


class WallRangeDialog(QDialog):
    def __init__(self, idx, ranges, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"W{idx+1} — Tilt Range")
        self.setMinimumWidth(260)
        dv   = QDoubleValidator(-89, 89, 2, self)
        form = QFormLayout(self)
        self.tmin = QLineEdit(str(ranges.get("tmin", 0.0))); self.tmin.setValidator(dv)
        self.tmax = QLineEdit(str(ranges.get("tmax", 0.0))); self.tmax.setValidator(dv)
        form.addRow("Tilt min [deg]:", self.tmin)
        form.addRow("Tilt max [deg]:", self.tmax)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_ranges(self):
        t1, t2 = float(self.tmin.text()), float(self.tmax.text())
        return {"tmin": min(t1, t2), "tmax": max(t1, t2)}


class HeightRangeDialog(QDialog):
    def __init__(self, ranges, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Z — Height Range")
        self.setMinimumWidth(260)
        dv   = QDoubleValidator(0.1, 100, 2, self)
        form = QFormLayout(self)
        self.zmin = QLineEdit(str(ranges.get("zmin", 0.0))); self.zmin.setValidator(dv)
        self.zmax = QLineEdit(str(ranges.get("zmax", 0.0))); self.zmax.setValidator(dv)
        form.addRow("Z min [m]:", self.zmin)
        form.addRow("Z max [m]:", self.zmax)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_ranges(self):
        return {"zmin": float(self.zmin.text()), "zmax": float(self.zmax.text())}


# ── Read-only canvas ──────────────────────────────────────────────────────────

_SYM_AXIS = ((0.0, 0.0), (0.0, 1.0))   # fixed symmetry axis: X = 0 (igual que room_design.py)


class OptCanvas(FigureCanvas):
    itemSelected = Signal(str, int)

    def __init__(self, parent=None):
        fig = Figure(facecolor="#1e1e1e")
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = fig.add_subplot(111)
        self._style_ax()
        self.vertices:   list = []
        self.wall_props: list = []
        self._sel_type = None
        self._sel_idx  = None
        self._sym_active = False
        self._locked_vertices: set = set()
        self.audience_verts: list = []
        self._audience_locked: set = set()
        self.source_positions: list = []
        self.mpl_connect("button_press_event", self._on_click)

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

    def load(self, vertices, wall_props, is_symmetric: bool = None, locked_vertices=None,
              audience_verts=None, audience_locked=None, source_positions=None):
        self.vertices   = list(vertices)
        self.wall_props = list(wall_props)
        self._sel_type  = self._sel_idx = None
        if is_symmetric is not None:
            self._sym_active = is_symmetric
        self._locked_vertices = set(locked_vertices) if locked_vertices is not None else set()
        self.audience_verts   = list(audience_verts) if audience_verts is not None else []
        self._audience_locked = set(audience_locked) if audience_locked is not None else set()
        self.source_positions  = [tuple(s) for s in source_positions] if source_positions else []
        self._redraw()

    def set_symmetric(self, active: bool):
        """Activa/desactiva el dibujo del eje de simetría (X = 0)."""
        self._sym_active = active
        self._redraw()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or len(self.vertices) < 2:
            return
        click = (event.xdata, event.ydata)
        vd    = [((click[0]-v[0])**2 + (click[1]-v[1])**2)**0.5 for v in self.vertices]
        vi    = int(np.argmin(vd))
        if vd[vi] < 0.3:
            self._sel_type, self._sel_idx = "vertex", vi
            self._redraw()
            self.itemSelected.emit("vertex", vi)
            return
        if self.audience_verts:
            ad = [((click[0]-v[0])**2 + (click[1]-v[1])**2)**0.5 for v in self.audience_verts]
            ai = int(np.argmin(ad))
            if ad[ai] < 0.3:
                self._sel_type, self._sel_idx = "audience", ai
                self._redraw()
                self.itemSelected.emit("audience", ai)
                return
        wi = nearest_wall(click, self.vertices, tol=0.4)
        if wi is not None:
            self._sel_type, self._sel_idx = "wall", wi
            self._redraw()
            self.itemSelected.emit("wall", wi)

    def _redraw(self):
        ax    = self.ax
        ax.cla()
        self._style_ax()
        verts = self.vertices
        n     = len(verts)
        if n == 0:
            self.draw_idle()
            return
        xs, ys = [v[0] for v in verts], [v[1] for v in verts]
        if n >= 3:
            from matplotlib.patches import Polygon as P
            ax.add_patch(P(list(zip(xs, ys)), closed=True,
                           facecolor="#2a3f54", edgecolor="#aaaaaa",
                           linewidth=1.5, alpha=0.6))
        for i in range(n):
            j      = (i+1) % n
            x1, y1 = verts[i]
            x2, y2 = verts[j]
            sel    = self._sel_type == "wall" and self._sel_idx == i
            ax.plot([x1, x2], [y1, y2],
                    color="#00bfff" if sel else "#aaaaaa",
                    linewidth=3 if sel else 1.5, zorder=3)
            l = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            t = self.wall_props[i]["tilt_deg"] if self.wall_props else 0
            ax.text((x1+x2)/2, (y1+y2)/2, f" W{i+1}  {l:.2f}m  {t}°",
                    color="#888888", fontsize=7, zorder=4)
        for i, (x, y) in enumerate(verts):
            sel    = self._sel_type == "vertex" and self._sel_idx == i
            locked = i in self._locked_vertices
            color  = "#ffdd00" if sel else ("#666666" if locked else "#ffffff")
            ax.scatter([x], [y], color=color, s=60 if sel else 30, zorder=5)
            ax.text(x, y, f" V{i+1}", color="#cccccc", fontsize=8, zorder=6)

        self._draw_audience()
        self._draw_source()

        all_x = list(xs) + [v[0] for v in self.audience_verts]
        all_y = list(ys) + [v[1] for v in self.audience_verts]
        for sp in self.source_positions:
            all_x.append(sp[0])
            all_y.append(sp[1])
        pad = 1.0
        ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
        ax.set_ylim(min(all_y)-pad, max(all_y)+pad)
        if self._sym_active:
            self._draw_sym_axis(ax.get_xlim(), ax.get_ylim())
        self.draw_idle()

    def _draw_audience(self):
        """Dibuja el audience_area, mismo estilo que room_design.py."""
        ax    = self.ax
        verts = self.audience_verts
        n     = len(verts)
        if n == 0:
            return
        if n >= 3:
            xs, ys = [v[0] for v in verts], [v[1] for v in verts]
            from matplotlib.patches import Polygon as P
            ax.add_patch(P(list(zip(xs, ys)), closed=True,
                           facecolor="#5a8f3c", edgecolor="#8fd14f",
                           linewidth=1.5, alpha=0.35, zorder=2))
        for i in range(n):
            j      = (i + 1) % n
            x1, y1 = verts[i]
            x2, y2 = verts[j]
            ax.plot([x1, x2], [y1, y2], color="#8fd14f", linewidth=1.3, zorder=3)
        for i, (x, y) in enumerate(verts):
            sel    = self._sel_type == "audience" and self._sel_idx == i
            locked = i in self._audience_locked
            color  = "#ffdd00" if sel else ("#666666" if locked else "#8fd14f")
            ax.scatter([x], [y], color=color, s=50 if sel else 40, marker="s", zorder=5)
            ax.text(x, y, f" A{i+1}", color=color, fontsize=7, zorder=6)

    def _draw_source(self):
        """Dibuja la(s) fuente(s) (source_positions), mismo estilo que room_design.py.
        Si hay más de una fuente, se numeran (Source 1, Source 2, ...)."""
        if not self.source_positions:
            return
        ax = self.ax
        multi = len(self.source_positions) > 1
        for i, sp in enumerate(self.source_positions):
            sx, sy = sp[0], sp[1]
            ax.scatter([sx], [sy], color="#ff9d00", s=90, marker="*",
                       edgecolor="#1e1e1e", linewidth=0.8, zorder=7)
            label = f" Source {i+1}" if multi else " Source"
            ax.text(sx, sy, label, color="#ff9d00", fontsize=8,
                    fontweight="bold", zorder=8)

    def _draw_sym_axis(self, xlim: tuple, ylim: tuple):
        """Dibuja el eje de simetría (X = 0), igual estilo que room_design.py."""
        a, b = _SYM_AXIS
        d    = np.array([b[0]-a[0], b[1]-a[1]], float)
        d   /= np.linalg.norm(d)
        diag = math.hypot(xlim[1]-xlim[0], ylim[1]-ylim[0])
        A    = np.array(a, float)
        p1, p2 = A - d*diag*2, A + d*diag*2
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                     color="#ff66cc", linestyle="--", linewidth=1.4, zorder=2)


# ── Table helpers ─────────────────────────────────────────────────────────────

def _make_table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setMaximumHeight(160)
    return t


def _add_row(table, label, v1, v2, v3="—", v4="—", enabled=True, locked=False):
    r  = table.rowCount()
    table.insertRow(r)
    cb = QCheckBox()
    cb.setChecked(enabled and not locked)
    cb.setEnabled(not locked)
    cb.setToolTip("Locked: mirrored vertex (X<0)" if locked else "Enable/disable for GA")
    w   = QWidget()
    lay = QHBoxLayout(w)
    lay.addWidget(cb)
    lay.setAlignment(Qt.AlignCenter)
    lay.setContentsMargins(0, 0, 0, 0)
    table.setCellWidget(r, 0, w)
    for c, val in enumerate([label, v1, v2, v3, v4], start=1):
        item = QTableWidgetItem(str(val))
        if val == "—" or locked:
            item.setFlags(Qt.ItemIsEnabled if val == "—" else Qt.NoItemFlags)
        table.setItem(r, c, item)
    return cb


def _update_row(table, row, v1, v2, v3="—", v4="—"):
    for c, val in enumerate([v1, v2, v3, v4], start=2):
        item = QTableWidgetItem(str(val))
        if val == "—":
            item.setFlags(Qt.ItemIsEnabled)
        table.setItem(row, c, item)


# ── Screen 1: GA configuration ────────────────────────────────────────────────

class GAConfigScreen(QWidget):
    runRequested = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state          = state
        self.output_dir     = ""
        self.vertex_ranges: list[dict]    = []
        self.wall_ranges:   list[dict]    = []
        self.height_ranges: dict          = {"zmin": 0.0, "zmax": 0.0}
        self.vertex_cbs:    list          = []
        self.wall_cbs:      list          = []
        self.height_cb      = None

        # Overlay created before _build so it exists when layout runs
        self._overlay = QLabel("Calculating...", self)
        self._overlay.setAlignment(Qt.AlignCenter)
        self._overlay.setStyleSheet("""
            background-color: rgba(0,0,0,200);
            color: white; font-size: 22pt; font-weight: bold; border-radius: 8px;
        """)
        self._overlay.hide()

        self._build()

        # Restaurar config previa si existe en el state
        self._sync_from_state()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        root.addWidget(self._build_canvas_panel(), stretch=3)
        root.addWidget(self._build_control_panel(), stretch=1)

    def _build_canvas_panel(self):
        box = QGroupBox(" Floor Plan (read-only)")
        lay = QVBoxLayout(box)
        self.canvas = OptCanvas()
        self.canvas.itemSelected.connect(self._on_item_selected)
        lay.addWidget(self.canvas)
        return box

    def _build_control_panel(self):
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Info label
        self.info_lbl = QLabel("Click a vertex or wall to set its GA range.")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet("color: #888888; font-size: 9pt;")
        lay.addWidget(self.info_lbl)

        # Output directory
        dir_box = QGroupBox(" Output Directory")
        dir_lay = QVBoxLayout(dir_box)

        dir_row = QHBoxLayout()
        self.dir_lbl = QLabel("Not set")
        self.dir_lbl.setStyleSheet("color: #888888; font-size: 8pt;")
        self.dir_lbl.setWordWrap(True)
        dir_row.addWidget(self.dir_lbl, stretch=1)
        btn_dir = QPushButton("Browse")
        btn_dir.setProperty("role", "secondary")
        btn_dir.setFixedWidth(70)
        btn_dir.clicked.connect(self._choose_output_dir)
        dir_row.addWidget(btn_dir)
        dir_lay.addLayout(dir_row)

        prefix_row = QHBoxLayout()
        prefix_row.addWidget(QLabel("File prefix:"))
        self.run_name_edit = QLineEdit("")
        self.run_name_edit.setPlaceholderText("e.g. room_v1")
        self.run_name_edit.textChanged.connect(self._sync_to_state)
        prefix_row.addWidget(self.run_name_edit)
        dir_lay.addLayout(prefix_row)

        lay.addWidget(dir_box)

        # GA run parameters
        ga_params_box = QGroupBox(" GA Parameters")
        ga_params_lay = QGridLayout(ga_params_box)
        ga_params_lay.addWidget(QLabel("Number of generations:"), 0, 0)
        self.n_generations_edit = QLineEdit("100")
        self.n_generations_edit.setValidator(QIntValidator(1, 100000, self))
        self.n_generations_edit.textChanged.connect(self._sync_to_state)
        ga_params_lay.addWidget(self.n_generations_edit, 0, 1)

        ga_params_lay.addWidget(QLabel("Rooms per generation:"), 1, 0)
        self.sol_per_pops_edit = QLineEdit("20")
        self.sol_per_pops_edit.setValidator(QIntValidator(1, 100000, self))
        self.sol_per_pops_edit.textChanged.connect(self._sync_to_state)
        ga_params_lay.addWidget(self.sol_per_pops_edit, 1, 1)
        lay.addWidget(ga_params_box)

        # Minimum distance between mics
        mmd_box = QGroupBox(" Minimum Mic Distance")
        mmd_lay = QHBoxLayout(mmd_box)
        mmd_lay.addWidget(QLabel("Min dist [m]:"))
        self.min_mic_distance_edit = QLineEdit("0.5")
        self.min_mic_distance_edit.setValidator(QDoubleValidator(0.01, 60.0, 3, self))
        mmd_lay.addWidget(self.min_mic_distance_edit)
        lay.addWidget(mmd_box)

        # Vertices table
        vbox = QGroupBox(" Vertices")
        vlay = QVBoxLayout(vbox)
        self.vtable = _make_table(["✓", "Param", "X min", "X max", "Y min", "Y max"])
        vlay.addWidget(self.vtable)
        lay.addWidget(vbox)

        # Wall tilt table
        wbox = QGroupBox(" Wall Tilt")
        wlay = QVBoxLayout(wbox)
        self.wtable = _make_table(["✓", "Param", "Tilt min", "Tilt max", "", ""])
        wlay.addWidget(self.wtable)
        lay.addWidget(wbox)

        # Height table
        hbox = QGroupBox(" Height")
        hlay = QVBoxLayout(hbox)
        self.htable = _make_table(["✓", "Param", "Min", "Max", "", ""])
        hlay.addWidget(self.htable)
        lay.addWidget(hbox)

        lay.addStretch()

        # Buttons
        btn_frame = QFrame()
        grid = QGridLayout(btn_frame)
        grid.setSpacing(6)
        grid.setContentsMargins(0, 6, 0, 6)
        for text, role, cb, r, c, cs in [
            ("Load from Design", "secondary", self._load_from_design, 0, 0, 1),
            ("Load Room File",   "secondary", self._load_room_file,   0, 1, 1),
            ("Clear Ranges",     "danger",    self._clear_ranges,     1, 0, 1),
            ("Optimize",         "success",   self._run_ga,           1, 1, 1),
        ]:
            btn = QPushButton(text)
            btn.setProperty("role", role)
            btn.clicked.connect(cb)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            grid.addWidget(btn, r, c, 1, cs)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        lay.addWidget(btn_frame)
        return panel

    # ── Overlay ───────────────────────────────────────────────────────────────
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

    # ── Output dir ────────────────────────────────────────────────────────────
    def _choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir = path
            self.dir_lbl.setText(path)
            self.dir_lbl.setStyleSheet("color: #aaaaaa; font-size: 8pt;")
            self._sync_to_state()

    # ── State sync ────────────────────────────────────────────────────────────
    def _sync_to_state(self):
        """Escribe los rangos actuales de las tablas en state.ga_config.

        Usamos zip() en vez de indexar self.vertex_cbs[i]/self.wall_cbs[i]
        directamente porque este método puede dispararse (via textChanged)
        desde _sync_from_state, en un punto en que self.vertex_ranges /
        self.wall_ranges ya están actualizados pero las listas de checkboxes
        todavía no se reconstruyeron (eso ocurre después, en _rebuild_tables).
        zip() simplemente trunca a la lista más corta en ese caso transitorio,
        evitando un IndexError."""
        cfg = self.state.ga_config
        cfg["vertex_ranges"] = [
            {**r, "enabled": cb.isChecked()}
            for r, cb in zip(self.vertex_ranges, self.vertex_cbs)
        ]
        cfg["wall_ranges"] = [
            {**r, "enabled": cb.isChecked()}
            for r, cb in zip(self.wall_ranges, self.wall_cbs)
        ]
        hr = self.height_ranges.copy()
        hr["enabled"] = self.height_cb.isChecked() if self.height_cb else True
        cfg["height_ranges"] = hr
        cfg["output_dir"]    = self.output_dir
        cfg["min_mic_distance"] = float(self.min_mic_distance_edit.text() or 0.5)
        cfg["run_name"]      = self.run_name_edit.text()
        cfg["n_generations"] = int(self.n_generations_edit.text() or 100)
        cfg["sol_per_pops"]  = int(self.sol_per_pops_edit.text() or 20)

    def _sync_from_state(self):
        """Lee state.ga_config y reconstruye las tablas si hay geometría disponible."""
        cfg  = self.state.ga_config
        geom = self.state.room_geometry.get("data", {})
        if not geom.get("vertices"):
            return

        verts  = [(v[0], v[1]) for v in geom["vertices"].values()]
        walls  = [{"id": f"W{i+1}", "tilt_deg": geom.get("walls", {}).get(f"W{i+1}", 0.0)}
                  for i in range(len(verts))]
        height = geom.get("Z", 3.0)
        audience_verts, source_positions = _extract_audience_and_source(geom)

        saved_vr = cfg.get("vertex_ranges", [])
        saved_ar = cfg.get("audience_ranges", [])
        saved_wr = cfg.get("wall_ranges",   [])
        saved_hr = cfg.get("height_ranges", {})

        sym = self.state.symmetric

        # Usar rangos guardados si coincide la cantidad de vértices/paredes;
        # si no, inicializar en cero.
        self.vertex_ranges = (saved_vr if len(saved_vr) == len(verts)
                              else [{"vertex": f"V{i+1}", "xmin": 0., "xmax": 0., "ymin": 0., "ymax": 0.}
                                    for i in range(len(verts))])
        locked_set = set()
        for i, r in enumerate(self.vertex_ranges):
            r.setdefault("vertex", f"V{i+1}")
            locked = sym and verts[i][0] < 0
            r["locked"] = locked
            if locked:
                locked_set.add(i)

        self.audience_ranges = (saved_ar if len(saved_ar) == len(audience_verts)
                                else [{"vertex": f"A{i+1}", "xmin": 0., "xmax": 0., "ymin": 0., "ymax": 0.}
                                      for i in range(len(audience_verts))])
        audience_locked_set = set()
        for i, r in enumerate(self.audience_ranges):
            r.setdefault("vertex", f"A{i+1}")
            locked = sym and audience_verts[i][0] < 0
            r["locked"] = locked
            if locked:
                audience_locked_set.add(i)

        self.wall_ranges   = (saved_wr if len(saved_wr) == len(walls)
                              else [{"tmin": 0., "tmax": 0.} for _ in walls])
        self.height_ranges = (saved_hr if saved_hr
                              else {"zmin": height, "zmax": height})

        # Restaurar output dir, file prefix, min mic distance y parámetros de GA
        self.output_dir = cfg.get("output_dir", "")
        if self.output_dir:
            self.dir_lbl.setText(self.output_dir)
            self.dir_lbl.setStyleSheet("color: #aaaaaa; font-size: 8pt;")
        self.min_mic_distance_edit.setText(str(cfg.get("min_mic_distance", 0.5)))
        self.run_name_edit.setText(cfg.get("run_name", ""))
        self.n_generations_edit.setText(str(cfg.get("n_generations", 100)))
        self.sol_per_pops_edit.setText(str(cfg.get("sol_per_pops", 20)))

        self.canvas.load(verts, walls, sym, locked_set,
                          audience_verts, audience_locked_set, source_positions)
        self._rebuild_tables(verts, walls, height)

        # Restaurar estado de checkboxes
        for i, r in enumerate(self.vertex_ranges):
            if i < len(self.vertex_cbs):
                self.vertex_cbs[i].setChecked(r.get("enabled", True))
        for i, r in enumerate(self.wall_ranges):
            if i < len(self.wall_cbs):
                self.wall_cbs[i].setChecked(r.get("enabled", True))
        if self.height_cb:
            self.height_cb.setChecked(self.height_ranges.get("enabled", True))

    # ── Data loading ──────────────────────────────────────────────────────────
    def _load_room_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Room", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                raw = json.load(f)
            self.state.room_geometry = raw if "data" in raw else {"data": raw}
            self._load_from_design()
        except Exception as e:
            _show_error(self, f"Could not load file:\n{e}")

    def _load_from_design(self):
        geom = self.state.room_geometry.get("data", {})
        if not geom:
            _show_error(self, "No room data found.")
            return
        verts  = [(v[0], v[1]) for v in geom.get("vertices", {}).values()]
        walls  = [{"id": f"W{i+1}", "tilt_deg": geom.get("walls", {}).get(f"W{i+1}", 0.0)}
                  for i in range(len(verts))]
        height = geom.get("Z", 3.0)
        audience_verts, source_positions = _extract_audience_and_source(geom)
        sym = self.state.symmetric

        locked_set = {i for i, v in enumerate(verts) if sym and v[0] < 0}
        self.vertex_ranges = [{"vertex": f"V{i+1}", "xmin": 0., "xmax": 0., "ymin": 0., "ymax": 0.,
                                "locked": i in locked_set}
                               for i in range(len(verts))]

        audience_locked_set = {i for i, v in enumerate(audience_verts) if sym and v[0] < 0}
        self.audience_ranges = [{"vertex": f"A{i+1}", "xmin": 0., "xmax": 0., "ymin": 0., "ymax": 0.,
                                  "locked": i in audience_locked_set}
                                 for i in range(len(audience_verts))]

        self.wall_ranges   = [{"tmin": 0., "tmax": 0.} for _ in walls]
        self.height_ranges = {"zmin": height, "zmax": height}
        self.canvas.load(verts, walls, sym, locked_set,
                          audience_verts, audience_locked_set, source_positions)
        self._rebuild_tables(verts, walls, height)
        self._sync_to_state()

    def _rebuild_tables(self, verts, walls, height):
        self.vtable.setRowCount(0)
        self.wtable.setRowCount(0)
        self.htable.setRowCount(0)
        self.vertex_cbs.clear()
        self.wall_cbs.clear()
        for i, r in enumerate(self.vertex_ranges):
            locked = r.get("locked", False)
            self.vertex_cbs.append(
                _add_row(self.vtable, r["vertex"], r["xmin"], r["xmax"], r["ymin"], r["ymax"],
                         enabled=r.get("enabled", True), locked=locked))
        for i, r in enumerate(self.wall_ranges):
            self.wall_cbs.append(
                _add_row(self.wtable, f"W{i+1}", r["tmin"], r["tmax"]))
        hr = self.height_ranges
        self.height_cb = _add_row(self.htable, "Height", hr["zmin"], hr["zmax"])

    def _clear_ranges(self):
        n_v = len(self.vertex_ranges)
        self.vertex_ranges = [{"vertex": r.get("vertex", f"V{i+1}"),
                                "xmin": 0., "xmax": 0., "ymin": 0., "ymax": 0.,
                                "locked": r.get("locked", False)}
                               for i, r in enumerate(self.vertex_ranges)]
        self.wall_ranges   = [{"tmin": 0., "tmax": 0.} for _ in self.wall_ranges]
        self.height_ranges = {"zmin": 0., "zmax": 0.}
        for i in range(n_v):
            _update_row(self.vtable, i, 0.0, 0.0, 0.0, 0.0)
        for i in range(len(self.wall_ranges)):
            _update_row(self.wtable, i, 0.0, 0.0)
        if self.htable.rowCount() > 0:
            _update_row(self.htable, 0, 0.0, 0.0)
        self._sync_to_state()

    # ── Canvas selection ──────────────────────────────────────────────────────
    def _on_item_selected(self, kind, idx):
        if kind == "vertex" and idx < len(self.vertex_ranges):
            if self.vertex_ranges[idx].get("locked", False):
                self.info_lbl.setText(
                    f"{self.vertex_ranges[idx]['vertex']} is on the mirrored side (X<0) — "
                    f"edit its symmetric counterpart instead."
                )
                return
            self.info_lbl.setText(f"{self.vertex_ranges[idx]['vertex']} selected")
            self.vtable.selectRow(idx)
            dlg = VertexRangeDialog(idx, self.vertex_ranges[idx], self)
            if dlg.exec() == QDialog.Accepted:
                self.vertex_ranges[idx].update(dlg.get_ranges())
                r = self.vertex_ranges[idx]
                _update_row(self.vtable, idx, r["xmin"], r["xmax"], r["ymin"], r["ymax"])
                self._sync_to_state()
        elif kind == "wall" and idx < len(self.wall_ranges):
            self.info_lbl.setText(f"W{idx+1} selected")
            self.wtable.selectRow(idx)
            dlg = WallRangeDialog(idx, self.wall_ranges[idx], self)
            if dlg.exec() == QDialog.Accepted:
                self.wall_ranges[idx] = dlg.get_ranges()
                r = self.wall_ranges[idx]
                _update_row(self.wtable, idx, r["tmin"], r["tmax"])
                self._sync_to_state()

    def _run_ga(self):
        self.runRequested.emit()