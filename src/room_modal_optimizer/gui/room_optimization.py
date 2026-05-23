import json
import numpy as np

from geometry import nearest_wall
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QFrame, QDialog, QFormLayout,
    QDialogButtonBox, QFileDialog, QMessageBox,
    QCheckBox, QScrollArea, QAbstractItemView
)


# ── Dialogs de rangos ─────────────────────────────────────────────────────────

class VertexRangeDialog(QDialog):
    def __init__(self, idx: int, ranges: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"V{idx+1} — Range")
        self.setMinimumWidth(260)
        dv = QDoubleValidator(-999, 999, 3, self)
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

    def get_ranges(self) -> dict:
        return {
            "xmin": float(self.xmin.text()), "xmax": float(self.xmax.text()),
            "ymin": float(self.ymin.text()), "ymax": float(self.ymax.text()),
        }


class WallRangeDialog(QDialog):
    def __init__(self, idx: int, ranges: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"W{idx+1} — Tilt Range")
        self.setMinimumWidth(260)
        dv = QDoubleValidator(-89, 89, 2, self)
        form = QFormLayout(self)
        self.tmin = QLineEdit(str(ranges.get("tmin", 0.0))); self.tmin.setValidator(dv)
        self.tmax = QLineEdit(str(ranges.get("tmax", 0.0))); self.tmax.setValidator(dv)
        form.addRow("Tilt min [deg]:", self.tmin)
        form.addRow("Tilt max [deg]:", self.tmax)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_ranges(self) -> dict:
        tmin, tmax = float(self.tmin.text()), float(self.tmax.text())
        if tmin > tmax: tmin, tmax = tmax, tmin
        return {"tmin": tmin, "tmax": tmax}


class HeightRangeDialog(QDialog):
    def __init__(self, ranges: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Z — Height Range")
        self.setMinimumWidth(260)
        dv = QDoubleValidator(0.1, 100, 2, self)
        form = QFormLayout(self)
        self.zmin = QLineEdit(str(ranges.get("zmin", 0.0))); self.zmin.setValidator(dv)
        self.zmax = QLineEdit(str(ranges.get("zmax", 0.0))); self.zmax.setValidator(dv)
        form.addRow("Z min [m]:", self.zmin)
        form.addRow("Z max [m]:", self.zmax)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_ranges(self) -> dict:
        return {"zmin": float(self.zmin.text()), "zmax": float(self.zmax.text())}


# ── Canvas de solo lectura ────────────────────────────────────────────────────

class OptCanvas(FigureCanvas):
    itemSelected = Signal(str, int)

    def __init__(self, parent=None):
        self.fig = Figure(facecolor="#1e1e1e")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self._style_ax()
        self.vertices:   list[tuple[float, float]] = []
        self.wall_props: list[dict] = []
        self._sel_type = None
        self._sel_idx  = None
        self.mpl_connect("button_press_event", self._on_click)

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="#333333")

    def load(self, vertices, wall_props):
        self.vertices   = list(vertices)
        self.wall_props = list(wall_props)
        self._sel_type  = None
        self._sel_idx   = None
        self._redraw()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or len(self.vertices) < 2:
            return
        click = (event.xdata, event.ydata)
        vdists = [((click[0]-v[0])**2 + (click[1]-v[1])**2)**0.5 for v in self.vertices]
        vi = int(np.argmin(vdists))
        if vdists[vi] < 0.3:
            self._sel_type, self._sel_idx = "vertex", vi
            self._redraw()
            self.itemSelected.emit("vertex", vi)
            return
        wi = nearest_wall(click, self.vertices, tol=0.4)
        if wi is not None:
            self._sel_type, self._sel_idx = "wall", wi
            self._redraw()
            self.itemSelected.emit("wall", wi)

    def _redraw(self):
        ax = self.ax
        ax.cla()
        self._style_ax()
        verts = self.vertices
        n = len(verts)
        if n == 0:
            self.draw_idle()
            return
        xs, ys = [v[0] for v in verts], [v[1] for v in verts]
        if n >= 3:
            from matplotlib.patches import Polygon as MplPolygon
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True,
                         facecolor="#2a3f54", edgecolor="#aaaaaa", linewidth=1.5, alpha=0.6))
        for i in range(n):
            j = (i+1) % n
            x1, y1 = verts[i]; x2, y2 = verts[j]
            is_sel = (self._sel_type == "wall" and self._sel_idx == i)
            ax.plot([x1,x2],[y1,y2], color="#00bfff" if is_sel else "#aaaaaa",
                    linewidth=3 if is_sel else 1.5, zorder=3)
            mx, my = (x1+x2)/2, (y1+y2)/2
            length = ((x2-x1)**2+(y2-y1)**2)**0.5
            tilt = self.wall_props[i]["tilt_deg"] if self.wall_props else 0
            ax.text(mx, my, f" W{i+1}  {length:.2f}m  {tilt}°", color="#888888", fontsize=7, zorder=4)
        for i,(x,y) in enumerate(verts):
            is_sel = (self._sel_type == "vertex" and self._sel_idx == i)
            ax.scatter([x],[y], color="#ffdd00" if is_sel else "#ffffff",
                       s=60 if is_sel else 30, zorder=5)
            ax.text(x, y, f" V{i+1}", color="#cccccc", fontsize=8, zorder=6)
        pad = 1.0
        ax.set_xlim(min(xs)-pad, max(xs)+pad)
        ax.set_ylim(min(ys)-pad, max(ys)+pad)
        self.draw_idle()


# ── Tabla con checkbox ────────────────────────────────────────────────────────

def _make_table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setMaximumHeight(180)
    return t


def _add_table_row(table: QTableWidget, label: str, v1, v2, v3="—", v4="—", enabled=True):
    r = table.rowCount()
    table.insertRow(r)

    cb = QCheckBox()
    cb.setChecked(enabled)
    cb.setToolTip("Enable/disable this parameter for GA")
    cb_widget = QWidget()
    cb_lay = QHBoxLayout(cb_widget)
    cb_lay.addWidget(cb)
    cb_lay.setAlignment(Qt.AlignCenter)
    cb_lay.setContentsMargins(0, 0, 0, 0)
    table.setCellWidget(r, 0, cb_widget)

    for c, val in enumerate([label, v1, v2, v3, v4], start=1):
        item = QTableWidgetItem(str(val))
        if val == "—":
            item.setFlags(Qt.ItemIsEnabled)
        table.setItem(r, c, item)
    return cb


def _update_table_row(table: QTableWidget, row: int, v1, v2, v3="—", v4="—"):
    for c, val in enumerate([v1, v2, v3, v4], start=2):
        item = QTableWidgetItem(str(val))
        if val == "—":
            item.setFlags(Qt.ItemIsEnabled)
        table.setItem(row, c, item)


# ── Tab principal ─────────────────────────────────────────────────────────────

class RoomOptimizationTab(QWidget):

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.vertex_ranges: list[dict] = []
        self.wall_ranges:   list[dict] = []
        self.height_ranges: dict = {"zmin": 0.0, "zmax": 0.0}
        self.vertex_cbs: list[QCheckBox] = []
        self.wall_cbs:   list[QCheckBox] = []
        self.height_cb:  QCheckBox | None = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        root.addWidget(self._build_canvas_panel(), stretch=3)
        root.addWidget(self._build_control_panel(), stretch=1)

    # ── Canvas ────────────────────────────────────────────────────────────────
    def _build_canvas_panel(self) -> QGroupBox:
        box = QGroupBox(" Floor Plan (read-only)")
        lay = QVBoxLayout(box)
        self.canvas = OptCanvas()
        self.canvas.itemSelected.connect(self._on_item_selected)
        lay.addWidget(self.canvas)
        return box

    # ── Panel control ─────────────────────────────────────────────────────────
    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Info selección
        self.info_lbl = QLabel("Click a vertex or wall to set its GA range.")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet("color: #888888; font-size: 9pt;")
        lay.addWidget(self.info_lbl)

        # Tabla Vertices
        vbox = QGroupBox(" Vertices")
        vlay = QVBoxLayout(vbox)
        self.vtable = _make_table(["✓", "Param", "X min", "X max", "Y min", "Y max"])
        vlay.addWidget(self.vtable)
        lay.addWidget(vbox)

        # Tabla Wall Tilt
        wbox = QGroupBox(" Wall Tilt")
        wlay = QVBoxLayout(wbox)
        self.wtable = _make_table(["✓", "Param", "Tilt min", "Tilt max", "", ""])
        wlay.addWidget(self.wtable)
        lay.addWidget(wbox)

        # Tabla Height / Area / Volume
        hbox = QGroupBox(" Height / Area / Volume")
        hlay = QVBoxLayout(hbox)
        self.htable = _make_table(["✓", "Param", "Min", "Max", "", ""])
        hlay.addWidget(self.htable)
        lay.addWidget(hbox)

        lay.addStretch()

        # Botones
        btn_frame = QFrame()
        btn_grid  = QGridLayout(btn_frame)
        btn_grid.setSpacing(6)
        btn_grid.setContentsMargins(0, 6, 0, 6)
        for text, role, cb, r, c, cs in [
            ("Load from Design", "secondary", self._load_from_design, 0, 0, 1),
            ("Load Room File",   "secondary", self._load_room_file,   0, 1, 1),
            ("Run GA",           "success",   self._run_ga,           1, 0, 1),
            ("Clear Ranges",     "danger",    self._clear_ranges,     1, 1, 1),
        ]:
            btn = QPushButton(text)
            btn.setProperty("role", role)
            btn.clicked.connect(cb)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn_grid.addWidget(btn, r, c, 1, cs)
        btn_grid.setColumnStretch(0, 1)
        btn_grid.setColumnStretch(1, 1)
        lay.addWidget(btn_frame)

        return panel

    # ── Carga de datos ────────────────────────────────────────────────────────
    def _load_room_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Room", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            geom = data.get("data", data)
            self.state.room_geometry = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in geom.items()
            }
            self._load_from_design()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")

    def _load_from_design(self):
        geom = self.state.room_geometry
        if not geom:
            QMessageBox.warning(self, "Warning", "No room data found.")
            return

        verts  = [(v[0], v[1]) for k, v in geom.items()
                  if k.startswith("V") and isinstance(v, tuple)]
        n      = len(verts)
        walls  = [{"id": f"W{i+1}", "tilt_deg": geom.get(f"W{i+1}", 0.0)} for i in range(n)]
        height = geom.get("Z", 3.0)

        self.vertex_ranges = [{"xmin":0.0,"xmax":0.0,"ymin":0.0,"ymax":0.0} for _ in verts]
        self.wall_ranges   = [{"tmin":0.0,"tmax":0.0} for _ in walls]
        self.height_ranges = {"zmin": height, "zmax": height}

        self.canvas.load(verts, walls)
        self._rebuild_tables(verts, walls, height)

    def _rebuild_tables(self, verts, walls, height):
        from geometry import signed_area
        self.vtable.setRowCount(0)
        self.wtable.setRowCount(0)
        self.htable.setRowCount(0)
        self.vertex_cbs.clear()
        self.wall_cbs.clear()

        for i, r in enumerate(self.vertex_ranges):
            cb = _add_table_row(self.vtable, f"V{i+1}", r["xmin"], r["xmax"], r["ymin"], r["ymax"])
            self.vertex_cbs.append(cb)

        for i, r in enumerate(self.wall_ranges):
            cb = _add_table_row(self.wtable, f"W{i+1}", r["tmin"], r["tmax"])
            self.wall_cbs.append(cb)

        area = abs(signed_area(verts)) if len(verts) >= 3 else 0.0
        vol  = area * height
        hr   = self.height_ranges
        self.height_cb = _add_table_row(self.htable, "Height", hr["zmin"], hr["zmax"])
        _add_table_row(self.htable, "Area",   f"{area:.3f}", "—", enabled=False)
        _add_table_row(self.htable, "Volume", f"{vol:.3f}",  "—", enabled=False)

    # ── Selección ─────────────────────────────────────────────────────────────
    def _on_item_selected(self, kind: str, idx: int):
        if kind == "vertex":
            if idx >= len(self.vertex_ranges):
                return
            self.info_lbl.setText(f"V{idx+1} selected — set X/Y range")
            self.vtable.selectRow(idx)
            dlg = VertexRangeDialog(idx, self.vertex_ranges[idx], self)
            if dlg.exec() == QDialog.Accepted:
                self.vertex_ranges[idx] = dlg.get_ranges()
                r = self.vertex_ranges[idx]
                _update_table_row(self.vtable, idx, r["xmin"], r["xmax"], r["ymin"], r["ymax"])
        elif kind == "wall":
            if idx >= len(self.wall_ranges):
                return
            self.info_lbl.setText(f"W{idx+1} selected — set tilt range")
            self.wtable.selectRow(idx)
            dlg = WallRangeDialog(idx, self.wall_ranges[idx], self)
            if dlg.exec() == QDialog.Accepted:
                self.wall_ranges[idx] = dlg.get_ranges()
                r = self.wall_ranges[idx]
                _update_table_row(self.wtable, idx, r["tmin"], r["tmax"])

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _run_ga(self):
        # Recopilar parámetros activos
        active_vertices = [
            {"id": f"V{i+1}", **self.vertex_ranges[i]}
            for i, cb in enumerate(self.vertex_cbs) if cb.isChecked()
        ]
        active_walls = [
            {"id": f"W{i+1}", **self.wall_ranges[i]}
            for i, cb in enumerate(self.wall_cbs) if cb.isChecked()
        ]
        optimize_height = self.height_cb.isChecked() if self.height_cb else False
        print("Running GA with:")
        print("  Vertices:", active_vertices)
        print("  Walls:",    active_walls)
        print("  Height:",   self.height_ranges if optimize_height else "disabled")

    def _clear_ranges(self):
        self.vertex_ranges = [{"xmin":0.0,"xmax":0.0,"ymin":0.0,"ymax":0.0}
                               for _ in self.vertex_ranges]
        self.wall_ranges   = [{"tmin":0.0,"tmax":0.0} for _ in self.wall_ranges]
        self.height_ranges = {"zmin":0.0,"zmax":0.0}
        for i in range(len(self.vertex_ranges)):
            _update_table_row(self.vtable, i, 0.0, 0.0, 0.0, 0.0)
        for i in range(len(self.wall_ranges)):
            _update_table_row(self.wtable, i, 0.0, 0.0)
        _update_table_row(self.htable, 0, 0.0, 0.0)