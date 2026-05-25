import json
import sys
import os
import numpy as np

from geometry import nearest_wall, signed_area, compute_ceiling
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QFrame, QDialog, QFormLayout,
    QDialogButtonBox, QFileDialog, QMessageBox,
    QCheckBox, QAbstractItemView, QStackedWidget,
    QComboBox, QApplication
)

sys.path.insert(0, os.path.dirname(__file__))
import dummy_functions as dumF


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


# ── Dialogs de rangos ─────────────────────────────────────────────────────────

class VertexRangeDialog(QDialog):
    def __init__(self, idx, ranges, parent=None):
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
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
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
        dv = QDoubleValidator(-89, 89, 2, self)
        form = QFormLayout(self)
        self.tmin = QLineEdit(str(ranges.get("tmin", 0.0))); self.tmin.setValidator(dv)
        self.tmax = QLineEdit(str(ranges.get("tmax", 0.0))); self.tmax.setValidator(dv)
        form.addRow("Tilt min [deg]:", self.tmin)
        form.addRow("Tilt max [deg]:", self.tmax)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_ranges(self):
        t1, t2 = float(self.tmin.text()), float(self.tmax.text())
        return {"tmin": min(t1, t2), "tmax": max(t1, t2)}


class HeightRangeDialog(QDialog):
    def __init__(self, ranges, parent=None):
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
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def get_ranges(self):
        return {"zmin": float(self.zmin.text()), "zmax": float(self.zmax.text())}


# ── Canvas de solo lectura ────────────────────────────────────────────────────

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
        self.mpl_connect("button_press_event", self._on_click)

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        for sp in ax.spines.values(): sp.set_edgecolor("#444444")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="#333333")

    def load(self, vertices, wall_props):
        self.vertices   = list(vertices)
        self.wall_props = list(wall_props)
        self._sel_type  = self._sel_idx = None
        self._redraw()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or len(self.vertices) < 2:
            return
        click = (event.xdata, event.ydata)
        vd = [((click[0]-v[0])**2 + (click[1]-v[1])**2)**0.5 for v in self.vertices]
        vi = int(np.argmin(vd))
        if vd[vi] < 0.3:
            self._sel_type, self._sel_idx = "vertex", vi
            self._redraw(); self.itemSelected.emit("vertex", vi); return
        wi = nearest_wall(click, self.vertices, tol=0.4)
        if wi is not None:
            self._sel_type, self._sel_idx = "wall", wi
            self._redraw(); self.itemSelected.emit("wall", wi)

    def _redraw(self):
        ax = self.ax; ax.cla(); self._style_ax()
        verts = self.vertices; n = len(verts)
        if n == 0: self.draw_idle(); return
        xs, ys = [v[0] for v in verts], [v[1] for v in verts]
        if n >= 3:
            from matplotlib.patches import Polygon as P
            ax.add_patch(P(list(zip(xs, ys)), closed=True,
                           facecolor="#2a3f54", edgecolor="#aaaaaa",
                           linewidth=1.5, alpha=0.6))
        for i in range(n):
            j = (i+1)%n; x1,y1 = verts[i]; x2,y2 = verts[j]
            sel = self._sel_type == "wall" and self._sel_idx == i
            ax.plot([x1,x2],[y1,y2], color="#00bfff" if sel else "#aaaaaa",
                    linewidth=3 if sel else 1.5, zorder=3)
            l = ((x2-x1)**2+(y2-y1)**2)**0.5
            t = self.wall_props[i]["tilt_deg"] if self.wall_props else 0
            ax.text((x1+x2)/2,(y1+y2)/2, f" W{i+1}  {l:.2f}m  {t}°",
                    color="#888888", fontsize=7, zorder=4)
        for i,(x,y) in enumerate(verts):
            sel = self._sel_type == "vertex" and self._sel_idx == i
            ax.scatter([x],[y], color="#ffdd00" if sel else "#ffffff",
                       s=60 if sel else 30, zorder=5)
            ax.text(x,y,f" V{i+1}", color="#cccccc", fontsize=8, zorder=6)
        pad = 1.0
        ax.set_xlim(min(xs)-pad, max(xs)+pad)
        ax.set_ylim(min(ys)-pad, max(ys)+pad)
        self.draw_idle()


# ── Tabla helpers ─────────────────────────────────────────────────────────────

def _make_table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setMaximumHeight(160)
    return t


def _add_row(table, label, v1, v2, v3="—", v4="—", enabled=True):
    r = table.rowCount(); table.insertRow(r)
    cb = QCheckBox(); cb.setChecked(enabled)
    cb.setToolTip("Enable/disable for GA")
    w = QWidget(); lay = QHBoxLayout(w)
    lay.addWidget(cb); lay.setAlignment(Qt.AlignCenter)
    lay.setContentsMargins(0,0,0,0)
    table.setCellWidget(r, 0, w)
    for c, val in enumerate([label, v1, v2, v3, v4], start=1):
        item = QTableWidgetItem(str(val))
        if val == "—": item.setFlags(Qt.ItemIsEnabled)
        table.setItem(r, c, item)
    return cb


def _update_row(table, row, v1, v2, v3="—", v4="—"):
    for c, val in enumerate([v1, v2, v3, v4], start=2):
        item = QTableWidgetItem(str(val))
        if val == "—": item.setFlags(Qt.ItemIsEnabled)
        table.setItem(row, c, item)


# ── Pantalla 1: configuración GA ──────────────────────────────────────────────

class GAConfigScreen(QWidget):
    runRequested = Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.vertex_ranges: list[dict] = []
        self.wall_ranges:   list[dict] = []
        self.height_ranges: dict = {"zmin": 0.0, "zmax": 0.0}
        self.vertex_cbs: list[QCheckBox] = []
        self.wall_cbs:   list[QCheckBox] = []
        self.height_cb = None
        self._build()

    def _build(self):
        # Overlay sobre toda la pantalla
        self._overlay = QLabel("Calculating...", self)
        self._overlay.setAlignment(Qt.AlignCenter)
        self._overlay.setStyleSheet("""
            background-color: rgba(0,0,0,200);
            color: white; font-size: 22pt; font-weight: bold; border-radius: 8px;
        """)
        self._overlay.hide()

        root = QHBoxLayout(self)
        root.setContentsMargins(8,8,8,8); root.setSpacing(10)
        root.addWidget(self._build_canvas_panel(), stretch=3)
        root.addWidget(self._build_control_panel(), stretch=1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.setGeometry(0, 0, self.width(), self.height())

    def show_overlay(self):
        self._overlay.show()
        self._overlay.raise_()
        QApplication.processEvents()

    def hide_overlay(self):
        self._overlay.hide()

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
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        self.info_lbl = QLabel("Click a vertex or wall to set its GA range.")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet("color: #888888; font-size: 9pt;")
        lay.addWidget(self.info_lbl)

        vbox = QGroupBox(" Vertices")
        vlay = QVBoxLayout(vbox)
        self.vtable = _make_table(["✓","Param","X min","X max","Y min","Y max"])
        vlay.addWidget(self.vtable); lay.addWidget(vbox)

        wbox = QGroupBox(" Wall Tilt")
        wlay = QVBoxLayout(wbox)
        self.wtable = _make_table(["✓","Param","Tilt min","Tilt max","",""])
        wlay.addWidget(self.wtable); lay.addWidget(wbox)

        hbox = QGroupBox(" Height / Area / Volume")
        hlay = QVBoxLayout(hbox)
        self.htable = _make_table(["✓","Param","Min","Max","",""])
        hlay.addWidget(self.htable); lay.addWidget(hbox)

        lay.addStretch()

        btn_frame = QFrame()
        grid = QGridLayout(btn_frame)
        grid.setSpacing(6); grid.setContentsMargins(0,6,0,6)
        for text, role, cb, r, c, cs in [
            ("Load from Design", "secondary", self._load_from_design, 0, 0, 1),
            ("Load Room File",   "secondary", self._load_room_file,   0, 1, 1),
            ("Clear Ranges",     "danger",    self._clear_ranges,     1, 0, 1),
            ("Optimize",         "success",   self._run_ga,           1, 1, 1),
        ]:
            btn = QPushButton(text); btn.setProperty("role", role)
            btn.clicked.connect(cb)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            grid.addWidget(btn, r, c, 1, cs)
        grid.setColumnStretch(0,1); grid.setColumnStretch(1,1)
        lay.addWidget(btn_frame)
        return panel

    def _load_room_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Room", "", "JSON Files (*.json)")
        if not path: return
        try:
            with open(path) as f: raw = json.load(f)
            self.state.room_geometry = raw if "data" in raw else {"data": raw}
            self._load_from_design()
        except Exception as e:
            _show_error(self, f"Could not load file:\n{e}")

    def _load_from_design(self):
        geom = self.state.room_geometry.get("data", {})
        if not geom:
            _show_error(self, "No room data found."); return
        verts  = [(v[0],v[1]) for v in geom.get("vertices",{}).values()]
        walls  = [{"id":f"W{i+1}","tilt_deg":geom.get("walls",{}).get(f"W{i+1}",0.0)}
                  for i in range(len(verts))]
        height = geom.get("Z", 3.0)
        self.vertex_ranges = [{"xmin":0.,"xmax":0.,"ymin":0.,"ymax":0.} for _ in verts]
        self.wall_ranges   = [{"tmin":0.,"tmax":0.} for _ in walls]
        self.height_ranges = {"zmin": height, "zmax": height}
        self.canvas.load(verts, walls)
        self._rebuild_tables(verts, walls, height)

    def _rebuild_tables(self, verts, walls, height):
        self.vtable.setRowCount(0)
        self.wtable.setRowCount(0)
        self.htable.setRowCount(0)
        self.vertex_cbs.clear(); self.wall_cbs.clear()
        for i, r in enumerate(self.vertex_ranges):
            self.vertex_cbs.append(_add_row(self.vtable, f"V{i+1}",
                r["xmin"], r["xmax"], r["ymin"], r["ymax"]))
        for i, r in enumerate(self.wall_ranges):
            self.wall_cbs.append(_add_row(self.wtable, f"W{i+1}", r["tmin"], r["tmax"]))
        area = abs(signed_area(verts)) if len(verts) >= 3 else 0.0
        hr = self.height_ranges
        self.height_cb = _add_row(self.htable, "Height", hr["zmin"], hr["zmax"])
        _add_row(self.htable, "Area",   f"{area:.3f}", "—", enabled=False)
        _add_row(self.htable, "Volume", f"{area*height:.3f}", "—", enabled=False)

    def _clear_ranges(self):
        n_v = len(self.vertex_ranges)
        self.vertex_ranges = [{"xmin":0.,"xmax":0.,"ymin":0.,"ymax":0.} for _ in self.vertex_ranges]
        self.wall_ranges   = [{"tmin":0.,"tmax":0.} for _ in self.wall_ranges]
        self.height_ranges = {"zmin":0.,"zmax":0.}
        for i in range(n_v): _update_row(self.vtable, i, 0.0, 0.0, 0.0, 0.0)
        for i in range(len(self.wall_ranges)): _update_row(self.wtable, i, 0.0, 0.0)
        if self.htable.rowCount() > 0: _update_row(self.htable, 0, 0.0, 0.0)

    def _on_item_selected(self, kind, idx):
        if kind == "vertex" and idx < len(self.vertex_ranges):
            self.info_lbl.setText(f"V{idx+1} selected")
            self.vtable.selectRow(idx)
            dlg = VertexRangeDialog(idx, self.vertex_ranges[idx], self)
            if dlg.exec() == QDialog.Accepted:
                self.vertex_ranges[idx] = dlg.get_ranges()
                r = self.vertex_ranges[idx]
                _update_row(self.vtable, idx, r["xmin"], r["xmax"], r["ymin"], r["ymax"])
        elif kind == "wall" and idx < len(self.wall_ranges):
            self.info_lbl.setText(f"W{idx+1} selected")
            self.wtable.selectRow(idx)
            dlg = WallRangeDialog(idx, self.wall_ranges[idx], self)
            if dlg.exec() == QDialog.Accepted:
                self.wall_ranges[idx] = dlg.get_ranges()
                r = self.wall_ranges[idx]
                _update_row(self.wtable, idx, r["tmin"], r["tmax"])

    def _run_ga(self):
        self.runRequested.emit()


# ── Pantalla 2: resultados modales ────────────────────────────────────────────

class ModalResultsScreen(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8,8,8,8); root.setSpacing(10)
        root.addWidget(self._build_views_panel(), stretch=3)
        root.addWidget(self._build_right_panel(), stretch=1)

    def _build_views_panel(self):
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(8)

        top = QHBoxLayout(); top.setSpacing(8)

        box2d = QGroupBox(" 2D Plant View")
        l2 = QVBoxLayout(box2d)
        self.canvas_2d = OptCanvas()
        l2.addWidget(self.canvas_2d)
        top.addWidget(box2d, stretch=1)

        box3d = QGroupBox(" 3D Room Viewer")
        l3 = QVBoxLayout(box3d)
        self.fig3d     = Figure(facecolor="#000000")
        self.canvas_3d = FigureCanvas(self.fig3d)
        self.canvas_3d.setStyleSheet("background-color: #000000;")
        self.canvas_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        l3.addWidget(self.canvas_3d)
        top.addWidget(box3d, stretch=1)

        lay.addLayout(top, stretch=2)

        hist_box = QGroupBox(" Room Modal Footprint")
        hist_lay = QVBoxLayout(hist_box)
        self.fig_hist    = Figure(facecolor="#1e1e1e")
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.canvas_hist.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hist_lay.addWidget(self.canvas_hist)
        self.freq_lbl = QLabel("")
        self.freq_lbl.setAlignment(Qt.AlignCenter)
        self.freq_lbl.setStyleSheet("color: #00bfff; font-size: 10pt; font-weight: bold;")
        hist_lay.addWidget(self.freq_lbl)
        lay.addWidget(hist_box, stretch=1)

        return panel

    def _build_right_panel(self):
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(8)

        self.room_combo = QComboBox()
        self.room_combo.addItem("Base Room")
        self.room_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self.room_combo)

        data_box = QGroupBox(" Room Data")
        data_lay = QVBoxLayout(data_box)
        self.room_data_lbl = QLabel("No data loaded.")
        self.room_data_lbl.setWordWrap(True)
        self.room_data_lbl.setAlignment(Qt.AlignTop)
        self.room_data_lbl.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        data_lay.addWidget(self.room_data_lbl)
        self.fsi_lbl = QLabel("")
        self.fsi_lbl.setWordWrap(True)
        self.fsi_lbl.setAlignment(Qt.AlignTop)
        self.fsi_lbl.setStyleSheet("""
            color: #00bfff; font-size: 10pt; font-weight: bold;
            border-top: 1px solid #444444; padding-top: 6px; margin-top: 4px;
        """)
        data_lay.addWidget(self.fsi_lbl)
        lay.addWidget(data_box, stretch=2)

        opts_box = QGroupBox(" Options")
        opts_lay = QVBoxLayout(opts_box)
        btn_back = QPushButton("← Back to GA Config")
        btn_back.setProperty("role", "secondary")
        btn_back.clicked.connect(self._go_back)
        opts_lay.addWidget(btn_back)
        lay.addWidget(opts_box, stretch=1)

        return panel

    def load_room(self, geom: dict):
        try:
            verts = [(v[0],v[1]) for v in geom.get("vertices",{}).values()]
            walls = [{"id":f"W{i+1}","tilt_deg":geom.get("walls",{}).get(f"W{i+1}",0.0)}
                     for i in range(len(verts))]
            self.canvas_2d.load(verts, walls)
            lines = [f"Z: {geom.get('Z','')} m"]
            for k,v in geom.get("vertices",{}).items():
                lines.append(f"{k}: ({v[0]:.3f}, {v[1]:.3f})")
            for k,v in geom.get("walls",{}).items():
                lines.append(f"{k}: {v}°")
            self.room_data_lbl.setText("\n".join(lines))
            self.fsi_lbl.setText("")
            self._render_3d(geom)
        except Exception as e:
            _show_error(self, f"Error loading room:\n{e}")

    def run_simulation(self):
        try:
            freqs, fsi = dumF.modal_sim(self.state.room_geometry)
            self.fsi_lbl.setText(f"FSI: {fsi:.4f}" if isinstance(fsi, float) else f"FSI: {fsi}")
            self._plot_histogram(np.array(freqs))
        except Exception as e:
            self.fsi_lbl.setText("FSI: —")
            _show_error(self, f"Simulation error:\n{e}")

    def _plot_histogram(self, freqs: np.ndarray):
        try:
            self.fig_hist.clear()
            self.fig_hist.set_facecolor("#1e1e1e")
            ax = self.fig_hist.add_subplot(111)
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#aaaaaa")
            ax.xaxis.label.set_color("#aaaaaa")
            ax.yaxis.label.set_color("#aaaaaa")
            for sp in ax.spines.values(): sp.set_edgecolor("#444444")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Count")
            n_bins = min(30, len(freqs))
            counts, edges, patches = ax.hist(freqs, bins=n_bins,
                                              color="#2a6496", edgecolor="#1a1a2e",
                                              rwidth=0.5)

            def on_pick(event):
                for i, p in enumerate(patches):
                    if p == event.artist:
                        lo, hi = edges[i], edges[i+1]
                        vals = freqs[(freqs >= lo) & (freqs < hi)]
                        self.freq_lbl.setText(
                            f"Bin {i+1}: {lo:.2f} – {hi:.2f} Hz  |  {len(vals)} modes")
                        for pp in patches: pp.set_facecolor("#2a6496")
                        p.set_facecolor("#00bfff")
                        self.canvas_hist.draw_idle()
                        break

            for p in patches: p.set_picker(True)
            self.fig_hist.canvas.mpl_connect("pick_event", on_pick)
            self.fig_hist.tight_layout()
            self.canvas_hist.draw()
        except Exception as e:
            _show_error(self, f"Error plotting histogram:\n{e}")

    def _render_3d(self, geom: dict):
        try:
            import pyvista as pv
            verts  = [(v[0],v[1]) for v in geom.get("vertices",{}).values()]
            tilts  = list(geom.get("walls",{}).values())
            n_v    = len(verts)
            tilts  = (tilts + [0.0]*n_v)[:n_v]
            height = geom.get("Z", 3.0)
            floor, ceiling = compute_ceiling(verts, height, tilts)
            n   = len(floor)
            pts = np.array([(x,y,0.0) for x,y in floor] + list(ceiling))
            faces = (
                [[n] + list(reversed(range(n)))] +
                [[n] + list(range(n, 2*n))] +
                [[4, i, (i+1)%n, (i+1)%n+n, i+n] for i in range(n)]
            )
            mesh    = pv.PolyData(pts, np.hstack(faces))
            plotter = pv.Plotter(off_screen=True)
            plotter.set_background("#000000")
            plotter.add_mesh(mesh, color="silver", show_edges=False, opacity=0.3)
            plotter.view_isometric(); plotter.reset_camera()
            screenshot = plotter.screenshot(); plotter.close()
            self.fig3d.clear(); self.fig3d.set_facecolor("#000000")
            ax = self.fig3d.add_subplot(111)
            ax.set_facecolor("#000000"); ax.imshow(screenshot); ax.axis("off")
            self.fig3d.tight_layout(pad=0); self.canvas_3d.draw()
        except Exception as e:
            _show_error(self, f"3D render error:\n{e}")

    def _go_back(self):
        stack = self.parent()
        if isinstance(stack, QStackedWidget):
            stack.setCurrentIndex(0)


# ── Tab principal ─────────────────────────────────────────────────────────────

class RoomOptimizationTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        self.stack          = QStackedWidget()
        self.ga_screen      = GAConfigScreen(self.state)
        self.results_screen = ModalResultsScreen(self.state)
        self.stack.addWidget(self.ga_screen)
        self.stack.addWidget(self.results_screen)
        self.stack.setCurrentWidget(self.ga_screen)
        self.ga_screen.runRequested.connect(self._on_run_requested)
        root.addWidget(self.stack)

    def _on_run_requested(self):
        # 1. Mostrar overlay sobre la pantalla de GA
        self.ga_screen.show_overlay()

        # 2. Cargar geometría en la pantalla de resultados
        geom_data = self.state.room_geometry.get("data", self.state.room_geometry)
        self.results_screen.load_room(geom_data)

        # 3. Correr simulación (bloqueante, overlay visible)
        self.results_screen.run_simulation()

        # 4. Ocultar overlay y cambiar pantalla
        self.ga_screen.hide_overlay()
        self.stack.setCurrentWidget(self.results_screen)

    def _load_from_design(self):
        self.ga_screen._load_from_design()