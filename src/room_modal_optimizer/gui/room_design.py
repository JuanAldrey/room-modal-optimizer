import json
import numpy as np

from geometry import compute_ceiling, nearest_wall, build_geometry_dict, signed_area
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QFormLayout, QDialogButtonBox,
    QSizePolicy, QFrame, QFileDialog, QMessageBox
)


# ── Dialog: configuración de pared ───────────────────────────────────────────

class WallConfigDialog(QDialog):
    def __init__(self, wall: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Configure {wall['id']}")
        self.setMinimumWidth(280)
        self.wall = dict(wall)

        form = QFormLayout(self)
        self.tilt = QLineEdit(str(self.wall["tilt_deg"]))
        self.tilt.setValidator(QDoubleValidator(-89, 89, 2, self))
        form.addRow("Inclination [deg]:", self.tilt)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _accept(self):
        try:
            self.wall["tilt_deg"] = float(self.tilt.text())
        except ValueError:
            return
        self.accept()

    def get_wall(self) -> dict:
        return self.wall


# ── Dialog: insertar vértice ──────────────────────────────────────────────────

class InsertVertexDialog(QDialog):
    def __init__(self, n: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Insert New Vertex")
        self.setMinimumWidth(240)
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel(
            f"Current vertices: 1 – {n}\n"
            f"Insert position (1 = before V1, {n+1} = after V{n}):"
        ))
        self.edit = QLineEdit("1")
        self.edit.setValidator(QIntValidator(1, n + 1, self))
        lay.addWidget(self.edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_index(self) -> int:
        return int(self.edit.text()) - 1  # 0-based


# ── Canvas 2D interactivo ─────────────────────────────────────────────────────

class FloorCanvas(FigureCanvas):
    verticesChanged = Signal()
    wallClicked     = Signal(int)

    def __init__(self, parent=None):
        fig = Figure(facecolor="#1e1e1e")
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax = fig.add_subplot(111)
        self._style_ax()

        self.vertices:   list[tuple[float, float]] = []
        self.wall_props: list[dict] = []
        self._undo_stack: list[list] = []
        self._redo_stack: list[list] = []
        self._mode      = "add"   # "add" | "select" | "place"
        self._place_idx  = None
        self._sel_wall   = None

        self.mpl_connect("button_press_event", self._on_click)

    # ── Estilo ────────────────────────────────────────────────────────────────
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
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 8)

    # ── Historial ─────────────────────────────────────────────────────────────
    def _save_undo(self):
        self._undo_stack.append(list(self.vertices))
        self._redo_stack.clear()

    def _restore(self, verts: list):
        self.vertices = list(verts)
        self._place_idx = None
        self._mode = "add"
        self._rebuild_wall_props()
        self._redraw()
        self.verticesChanged.emit()

    def undo(self):
        if self._undo_stack:
            self._redo_stack.append(list(self.vertices))
            self._restore(self._undo_stack.pop())

    def redo(self):
        if self._redo_stack:
            self._undo_stack.append(list(self.vertices))
            self._restore(self._redo_stack.pop())

    # ── Mutaciones ────────────────────────────────────────────────────────────
    def set_mode(self, mode: str):
        self._mode = mode
        self._place_idx = None
        self._sel_wall  = None
        self._redraw()

    def set_vertices(self, verts: list[tuple[float, float]]):
        self._save_undo()
        self.vertices = list(verts)
        self._rebuild_wall_props()
        self._redraw()
        self.verticesChanged.emit()

    def insert_vertex(self, idx: int):
        self._save_undo()
        self.vertices.insert(idx, (0.0, 0.0))
        self._rebuild_wall_props()
        self._place_idx = idx
        self._mode = "place"
        self._redraw()
        self.verticesChanged.emit()

    def clear(self):
        self._save_undo()
        self.vertices.clear()
        self.wall_props.clear()
        self._sel_wall  = None
        self._place_idx = None
        self._redraw()
        self.verticesChanged.emit()

    def update_wall_tilt(self, idx: int, tilt: float):
        if 0 <= idx < len(self.wall_props):
            self.wall_props[idx]["tilt_deg"] = tilt
            self._redraw()

    # ── Eventos ───────────────────────────────────────────────────────────────
    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        if self._mode == "place" and event.button == 1 and self._place_idx is not None:
            self.vertices[self._place_idx] = (round(event.xdata, 3), round(event.ydata, 3))
            self._place_idx = None
            self._mode = "add"
            self._rebuild_wall_props()
            self._redraw()
            self.verticesChanged.emit()

        elif self._mode == "add" and event.button == 1:
            self._save_undo()
            self.vertices.append((round(event.xdata, 3), round(event.ydata, 3)))
            self._rebuild_wall_props()
            self._redraw()
            self.verticesChanged.emit()

        elif self._mode == "select" and event.button == 1 and len(self.vertices) >= 2:
            idx = nearest_wall((event.xdata, event.ydata), self.vertices)
            if idx is not None:
                self._sel_wall = idx
                self._redraw()
                self.wallClicked.emit(idx)

    # ── Wall props ────────────────────────────────────────────────────────────
    def _rebuild_wall_props(self):
        n   = len(self.vertices)
        old = {w["id"]: w for w in self.wall_props}
        self.wall_props = [
            old.get(f"W{i+1}", {
                "id": f"W{i+1}", "tilt_deg": 0.0,
                "locked": False, "optimize_tilt": False,
                "tilt_min": 0.0, "tilt_max": 0.0,
            })
            for i in range(n)
        ]

    # ── Redibujado ────────────────────────────────────────────────────────────
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
            ax.add_patch(MplPolygon(list(zip(xs, ys)), closed=True,
                facecolor="#2a3f54", edgecolor="#aaaaaa", linewidth=1.5, alpha=0.6))

        for i in range(n):
            j  = (i + 1) % n
            x1, y1 = verts[i]
            x2, y2 = verts[j]
            sel = (self._sel_wall == i)
            ax.plot([x1, x2], [y1, y2],
                    color="#00bfff" if sel else "#aaaaaa",
                    linewidth=3 if sel else 1.5, zorder=3)
            mx, my = (x1+x2)/2, (y1+y2)/2
            length = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            tilt   = self.wall_props[i]["tilt_deg"] if self.wall_props else 0
            ax.text(mx, my, f" W{i+1}  {length:.2f}m  {tilt}°",
                    color="#888888", fontsize=7, zorder=4)

        for i, (x, y) in enumerate(verts):
            placing = (self._mode == "place" and i == self._place_idx)
            ax.scatter([x], [y], color="#ffdd00" if placing else "#ffffff",
                       s=60 if placing else 30, zorder=5)
            label = f" V{i+1} ← click to place" if placing else f" V{i+1}"
            ax.text(x, y, label, color="#ffdd00" if placing else "#cccccc",
                    fontsize=8, zorder=6)

        pad = 1.0
        ax.set_xlim(min(xs)-pad, max(xs)+pad)
        ax.set_ylim(min(ys)-pad, max(ys)+pad)
        self.draw_idle()


# ── Tabla de vértices ─────────────────────────────────────────────────────────

class VertexTable(QTableWidget):
    verticesEdited = Signal(list)

    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["#", "X [m]", "Y [m]"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self._updating = False
        self.itemChanged.connect(self._on_edit)

    def load(self, verts: list[tuple[float, float]]):
        self._updating = True
        self.setRowCount(len(verts))
        for i, (x, y) in enumerate(verts):
            self.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.item(i, 0).setFlags(Qt.ItemIsEnabled)
            self.setItem(i, 1, QTableWidgetItem(str(x)))
            self.setItem(i, 2, QTableWidgetItem(str(y)))
        self._updating = False

    def _on_edit(self, item):
        if self._updating:
            return
        verts = []
        for r in range(self.rowCount()):
            try:
                x = float(self.item(r, 1).text())
                y = float(self.item(r, 2).text())
                verts.append((x, y))
            except (ValueError, AttributeError):
                return
        self.verticesEdited.emit(verts)


# ── Tab Room Design ───────────────────────────────────────────────────────────

class RoomDesignTab(QWidget):

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        root.addWidget(self._build_canvas_panel(), stretch=3)
        root.addWidget(self._build_control_panel(), stretch=1)

    # ── Canvas panel ─────────────────────────────────────────────────────────
    def _build_canvas_panel(self) -> QGroupBox:
        box = QGroupBox(" Floor Plan")
        lay = QVBoxLayout(box)
        self.canvas = FloorCanvas()
        self.canvas.verticesChanged.connect(self._on_verts_changed)
        self.canvas.wallClicked.connect(self._on_wall_clicked)
        lay.addWidget(self.canvas)
        return box

    # ── Control panel ─────────────────────────────────────────────────────────
    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Mode
        mode_box = QGroupBox(" Mode")
        mode_lay = QHBoxLayout(mode_box)
        self.btn_add = self._btn("Add Vertices", "success",   lambda: self._set_mode("add"))
        self.btn_sel = self._btn("Select Wall",  "secondary", lambda: self._set_mode("select"))
        self.btn_add.setCheckable(True); self.btn_add.setChecked(True)
        self.btn_sel.setCheckable(True)
        mode_lay.addWidget(self.btn_add)
        mode_lay.addWidget(self.btn_sel)
        lay.addWidget(mode_box)

        # Room info
        info_box = QGroupBox(" Room Info")
        info_grid = QGridLayout(info_box)
        info_grid.addWidget(QLabel("Floor area:"), 0, 0)
        self.area_lbl = QLabel("— m²")
        info_grid.addWidget(self.area_lbl, 0, 1)
        info_grid.addWidget(QLabel("Volume:"), 1, 0)
        self.vol_lbl = QLabel("— m³")
        info_grid.addWidget(self.vol_lbl, 1, 1)
        lay.addWidget(info_box)

        # Height
        h_box = QGroupBox(" Room Height [m]")
        h_lay = QHBoxLayout(h_box)
        self.height_edit = QLineEdit("3.0")
        self.height_edit.setValidator(QDoubleValidator(0.1, 100, 2, self))
        self.height_edit.textChanged.connect(self._update_room_info)
        h_lay.addWidget(self.height_edit)
        lay.addWidget(h_box)

        # Vertex table
        vt_box = QGroupBox(" Vertices")
        vt_lay = QVBoxLayout(vt_box)
        self.vtable = VertexTable()
        self.vtable.verticesEdited.connect(self._on_table_edit)
        vt_lay.addWidget(self.vtable)
        lay.addWidget(vt_box)

        # Buttons
        btn_frame = QFrame()
        btn_grid  = QGridLayout(btn_frame)
        btn_grid.setSpacing(6)
        btn_grid.setContentsMargins(0, 4, 0, 4)

        actions = [
            ("Undo",             "secondary", self.canvas.undo,       0, 0, 1),
            ("Redo",             "secondary", self.canvas.redo,       0, 1, 1),
            ("Clear",            "danger",    self.canvas.clear,      1, 0, 1),
            ("Preview 3D",       "secondary", self._preview_3d,       1, 1, 1),
            ("New Vertex",       "secondary", self._insert_vertex,    2, 0, 1),
            ("Load Room",        "secondary", self._load_room_file,   2, 1, 1),
            ("Save Room",        "success",   self._save_room_file,   3, 0, 1),
            ("To Room Optimize", "success",   self._to_room_optimize, 3, 1, 1),
        ]
        for text, role, cb, r, c, cs in actions:
            b = self._btn(text, role, cb)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn_grid.addWidget(b, r, c, 1, cs)

        btn_grid.setColumnStretch(0, 1)
        btn_grid.setColumnStretch(1, 1)
        lay.addWidget(btn_frame)
        lay.addStretch()

        return panel

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _btn(self, text, role, cb) -> QPushButton:
        btn = QPushButton(text)
        btn.setProperty("role", role)
        btn.clicked.connect(cb)
        return btn

    def _set_mode(self, mode: str):
        self.canvas.set_mode(mode)
        self.btn_add.setChecked(mode == "add")
        self.btn_sel.setChecked(mode == "select")

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_verts_changed(self):
        self.vtable.load(self.canvas.vertices)
        self._update_room_info()

    def _on_table_edit(self, verts: list):
        self.canvas.set_vertices(verts)

    def _on_wall_clicked(self, idx: int):
        dlg = WallConfigDialog(self.canvas.wall_props[idx], self)
        if dlg.exec() == QDialog.Accepted:
            self.canvas.wall_props[idx] = dlg.get_wall()
            self.canvas.update_wall_tilt(idx, dlg.get_wall()["tilt_deg"])

    def _update_room_info(self):
        verts = self.canvas.vertices
        if len(verts) < 3:
            self.area_lbl.setText("— m²")
            self.vol_lbl.setText("— m³")
            return
        try:
            area   = abs(signed_area(verts))
            height = float(self.height_edit.text() or 0)
            self.area_lbl.setText(f"{area:.3f} m²")
            self.vol_lbl.setText(f"{area * height:.3f} m³")
        except ValueError:
            self.area_lbl.setText("— m²")
            self.vol_lbl.setText("— m³")

    def _insert_vertex(self):
        n = len(self.canvas.vertices)
        dlg = InsertVertexDialog(n, self)
        if dlg.exec() == QDialog.Accepted:
            self.canvas.insert_vertex(dlg.get_index())

    # ── Load / Save ───────────────────────────────────────────────────────────
    def _load_room_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Room", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
            self.state.room_geometry = raw.get("data", raw)
            self.refresh_from_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")

    def _save_room_file(self):
        verts = self.canvas.vertices
        if len(verts) < 3:
            QMessageBox.warning(self, "Warning", "At least 3 vertices required.")
            return
        try:
            height = float(self.height_edit.text())
            tilts  = [w["tilt_deg"] for w in self.canvas.wall_props]
            floor, _ = compute_ceiling(verts, height, tilts)
        except Exception as e:
            QMessageBox.critical(self, "Geometry Error", str(e))
            return

        self.state.room_geometry = build_geometry_dict(
            floor, self.canvas.wall_props, height, original_verts=verts)

        path, _ = QFileDialog.getSaveFileName(self, "Save Room", "", "JSON Files (*.json)")
        if not path:
            return
        if not path.lower().endswith('.json'):
            path += '.json'
        with open(path, 'w') as f:
            json.dump(self.state.room_geometry, f, indent=4)
        print(f"Saved: {path}")

    def _to_room_optimize(self):
        verts = self.canvas.vertices
        if len(verts) < 3:
            QMessageBox.warning(self, "Warning", "At least 3 vertices required.")
            return
        try:
            height = float(self.height_edit.text())
            tilts  = [w["tilt_deg"] for w in self.canvas.wall_props]
            floor, _ = compute_ceiling(verts, height, tilts)
        except Exception as e:
            QMessageBox.critical(self, "Geometry Error", str(e))
            return

        self.state.room_geometry = build_geometry_dict(
            floor, self.canvas.wall_props, height, original_verts=verts)

        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "tabs"):
                opt_tab = parent.tabs.widget(1)
                if hasattr(opt_tab, "_load_from_design"):
                    opt_tab._load_from_design()
                    parent.tabs.setCurrentIndex(1)
                break
            parent = parent.parent()

    def refresh_from_state(self):
        geom = self.state.room_geometry.get("data", {})
        if not geom:
            return
        verts_data = geom.get("vertices", {})
        walls_data = geom.get("walls", {})
        height     = geom.get("Z", 3.0)
        verts      = [(v[0], v[1]) for v in verts_data.values()]
        n          = len(verts)
        wall_props = [{
            "id": f"W{i+1}", "tilt_deg": walls_data.get(f"W{i+1}", 0.0),
            "locked": False, "optimize_tilt": False,
            "tilt_min": 0.0, "tilt_max": 0.0,
        } for i in range(n)]

        self.height_edit.setText(str(height))
        self.canvas._undo_stack.clear()
        self.canvas._redo_stack.clear()
        self.canvas.vertices   = verts
        self.canvas.wall_props = wall_props
        self.canvas._redraw()
        self.vtable.load(verts)
        self._update_room_info()

    # ── Preview 3D ────────────────────────────────────────────────────────────
    def _preview_3d(self):
        verts = self.canvas.vertices
        if len(verts) < 3:
            return
        try:
            height = float(self.height_edit.text())
            tilts  = [w["tilt_deg"] for w in self.canvas.wall_props]
            floor, ceiling = compute_ceiling(verts, height, tilts)

            import pyvista as pv
            n   = len(floor)
            pts = np.array([(x, y, 0.0) for x, y in floor] + list(ceiling))
            faces = (
                [[n] + list(reversed(range(n)))] +
                [[n] + list(range(n, 2*n))] +
                [[4, i, (i+1)%n, (i+1)%n + n, i + n] for i in range(n)]
            )
            mesh = pv.PolyData(pts, np.hstack(faces))
            pl   = pv.Plotter(off_screen=False)
            pl.set_background("#1e1e1e")
            pl.add_mesh(mesh, show_edges=True, color="silver", opacity=0.5)
            pl.add_axes()
            pl.show()
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))