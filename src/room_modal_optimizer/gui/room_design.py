import numpy as np

from geometry import compute_ceiling, nearest_wall, build_geometry_dict
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QFormLayout, QCheckBox, QDialogButtonBox,
    QSizePolicy, QFrame, QSpacerItem
)


# ── Dialog de configuración de pared ─────────────────────────────────────────

class WallConfigDialog(QDialog):
    def __init__(self, wall: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Configure {wall['id']}")
        self.setMinimumWidth(320)
        self.wall = dict(wall)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        form = QFormLayout()
        dv = QDoubleValidator(-89, 89, 2, self)

        self.tilt_edit = QLineEdit(str(self.wall["tilt_deg"]))
        self.tilt_edit.setValidator(dv)
        form.addRow("Inclination [deg]:", self.tilt_edit)
        lay.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _accept(self):
        try:
            tilt = float(self.tilt_edit.text())
        except ValueError:
            return
        self.wall.update({"tilt_deg": tilt})
        self.accept()

    def get_wall(self):
        return self.wall


# ── Dialog para elegir ID de inserción ───────────────────────────────────────

class InsertVertexDialog(QDialog):
    def __init__(self, n_verts: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Insert New Vertex")
        self.setMinimumWidth(240)
        lay = QVBoxLayout(self)

        info = QLabel(f"Current vertices: 1 – {n_verts}\n"
                      f"Choose insert position (1 = before V1, {n_verts+1} = after V{n_verts}):")
        info.setWordWrap(True)
        lay.addWidget(info)

        self.spin = QLineEdit("1")
        self.spin.setValidator(QIntValidator(1, n_verts + 1, self))
        lay.addWidget(self.spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_index(self) -> int:
        return int(self.spin.text()) - 1  # 0-based


# ── Canvas 2D ─────────────────────────────────────────────────────────────────

class FloorCanvas(FigureCanvas):
    """Canvas matplotlib embebido para dibujar/editar la planta."""

    verticesChanged = Signal()
    wallClicked     = Signal(int)   # índice de pared

    def __init__(self, parent=None):
        self.fig = Figure(facecolor="#1e1e1e")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax = self.fig.add_subplot(111)
        self._style_ax()

        self.vertices: list[tuple[float,float]] = []
        self._redo_stack: list[tuple[float,float]] = []
        self.wall_props: list[dict] = []
        self._mode = "add"        # "add" | "select" | "place"
        self._place_idx = None    # índice del vértice a posicionar
        self._selected_wall = None

        self._poly_patch = None
        self._sel_line   = None

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
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 8)

    def insert_vertex(self, idx: int):
        """Inserta un vértice en (0,0) en la posición idx y entra en modo place."""
        self._redo_stack.clear()
        self.vertices.insert(idx, (0.0, 0.0))
        self._rebuild_wall_props()
        self._place_idx = idx
        self._mode = "place"
        self._redraw()
        self.verticesChanged.emit()

    def set_mode(self, mode: str):
        self._mode = mode
        self._place_idx = None
        self._selected_wall = None
        self._redraw()

    def set_vertices(self, verts: list[tuple[float,float]]):
        self.vertices = list(verts)
        self._rebuild_wall_props()
        self._redraw()
        self.verticesChanged.emit()

    def _rebuild_wall_props(self):
        n = len(self.vertices)
        old = {w["id"]: w for w in self.wall_props}
        self.wall_props = []
        for i in range(n):
            wid = f"W{i+1}"
            if wid in old:
                self.wall_props.append(old[wid])
            else:
                self.wall_props.append({
                    "id": wid,
                    "tilt_deg": 0.0, "locked": False,
                    "optimize_tilt": False,
                    "tilt_min": 0.0, "tilt_max": 0.0,
                })

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        if self._mode == "place" and event.button == 1:
            if self._place_idx is not None:
                x = round(float(event.xdata), 3)
                y = round(float(event.ydata), 3)
                self.vertices[self._place_idx] = (x, y)
                self._place_idx = None
                self._mode = "add"
                self._rebuild_wall_props()
                self._redraw()
                self.verticesChanged.emit()
            return

        if self._mode == "add" and event.button == 1:
            self.vertices.append((round(float(event.xdata),3),
                                  round(float(event.ydata),3)))
            self._rebuild_wall_props()
            self._redraw()
            self.verticesChanged.emit()

        elif self._mode == "select" and event.button == 1:
            if len(self.vertices) < 2:
                return
            idx = nearest_wall((event.xdata, event.ydata), self.vertices)
            if idx is not None:
                self._selected_wall = idx
                self._redraw()
                self.wallClicked.emit(idx)

    def undo(self):
        if self.vertices:
            self._redo_stack.append(self.vertices.pop())
            self._rebuild_wall_props()
            self._redraw()
            self.verticesChanged.emit()

    def redo(self):
        if self._redo_stack:
            self.vertices.append(self._redo_stack.pop())
            self._rebuild_wall_props()
            self._redraw()
            self.verticesChanged.emit()

    def clear(self):
        self._redo_stack.clear()
        self.vertices.clear()
        self.wall_props.clear()
        self._selected_wall = None
        self._redraw()
        self.verticesChanged.emit()

    def _redraw(self):
        ax = self.ax
        ax.cla()
        self._style_ax()

        verts = self.vertices
        n = len(verts)

        if n == 0:
            self.draw_idle()
            return

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]

        # Polígono relleno
        if n >= 3:
            patch = MplPolygon(list(zip(xs,ys)), closed=True,
                               facecolor="#2a3f54", edgecolor="#aaaaaa",
                               linewidth=1.5, alpha=0.6)
            ax.add_patch(patch)

        # Aristas
        for i in range(n):
            j = (i+1) % n if n >= 3 else None
            if j is None and i == n-1: break
            x1,y1 = verts[i]
            x2,y2 = verts[j] if j is not None else verts[i]
            if j is not None:
                color = "#00bfff" if i == self._selected_wall else "#aaaaaa"
                lw    = 3 if i == self._selected_wall else 1.5
                ax.plot([x1,x2],[y1,y2], color=color, linewidth=lw, zorder=3)
                # Etiqueta pared: longitud + ángulo
                mx, my = (x1+x2)/2, (y1+y2)/2
                length = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
                tilt   = self.wall_props[i]["tilt_deg"] if self.wall_props else 0
                ax.text(mx, my, f" W{i+1}  {length:.2f}m  {tilt}°",
                        color="#888888", fontsize=7, zorder=4)

        # Vértices
        ax.scatter(xs, ys, color="#ffffff", s=30, zorder=5)
        for i,(x,y) in enumerate(verts):
            is_placing = (self._mode == "place" and i == self._place_idx)
            color = "#ffdd00" if is_placing else "#cccccc"
            size  = 60 if is_placing else 30
            ax.scatter([x], [y], color=color, s=size, zorder=5)
            label = f" V{i+1} ← click to place" if is_placing else f" V{i+1}"
            ax.text(x, y, label, color=color, fontsize=8, zorder=6)

        # Ajustar límites
        pad = 1.0
        ax.set_xlim(min(xs)-pad, max(xs)+pad)
        ax.set_ylim(min(ys)-pad, max(ys)+pad)

        self.draw_idle()

    def update_wall_tilt(self, idx: int, tilt: float):
        if 0 <= idx < len(self.wall_props):
            self.wall_props[idx]["tilt_deg"] = tilt
            self._redraw()


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

    def load(self, verts: list[tuple[float,float]]):
        self._updating = True
        self.setRowCount(len(verts))
        for i,(x,y) in enumerate(verts):
            self.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.item(i,0).setFlags(Qt.ItemIsEnabled)
            self.setItem(i, 1, QTableWidgetItem(str(x)))
            self.setItem(i, 2, QTableWidgetItem(str(y)))
        self._updating = False

    def _on_edit(self, item):
        if self._updating: return
        verts = []
        for r in range(self.rowCount()):
            try:
                x = float(self.item(r,1).text())
                y = float(self.item(r,2).text())
                verts.append((x,y))
            except (ValueError, AttributeError):
                return
        self.verticesEdited.emit(verts)


# ── Tab principal Room Design ─────────────────────────────────────────────────

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

    # ── Panel canvas ──────────────────────────────────────────────────────────
    def refresh_from_state(self):
        """Carga el room_geometry del RoomState en el canvas."""
        geom = self.state.room_geometry
        if not geom:
            return
        verts = [(v[0], v[1]) for k, v in geom.items()
                 if k.startswith("V") and isinstance(v, tuple)]
        n = len(verts)
        wall_props = []
        for i in range(n):
            wall_props.append({
                "id": f"W{i+1}",
                "tilt_deg": geom.get(f"W{i+1}", 0.0),
                "locked": False,
                "optimize_tilt": False,
                "tilt_min": 0.0,
                "tilt_max": 0.0,
            })
        self.height_edit.setText(str(geom.get("Z", 3.0)))
        self.canvas.set_vertices(verts)
        self.canvas.wall_props = wall_props
        self.canvas._redraw()

    def _build_canvas_panel(self) -> QGroupBox:
        box = QGroupBox(" Floor Plan")
        lay = QVBoxLayout(box)

        self.canvas = FloorCanvas()
        self.canvas.verticesChanged.connect(self._on_verts_changed)
        self.canvas.wallClicked.connect(self._on_wall_clicked)
        lay.addWidget(self.canvas)

        return box

    # ── Panel control ─────────────────────────────────────────────────────────
    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Modo
        mode_box = QGroupBox(" Mode")
        mode_lay = QHBoxLayout(mode_box)
        self.btn_add = QPushButton("Add Vertices")
        self.btn_add.setProperty("role", "success")
        self.btn_add.setCheckable(True)
        self.btn_add.setChecked(True)
        self.btn_add.clicked.connect(lambda: self._set_mode("add"))

        self.btn_sel = QPushButton("Select Wall")
        self.btn_sel.setProperty("role", "secondary")
        self.btn_sel.setCheckable(True)
        self.btn_sel.clicked.connect(lambda: self._set_mode("select"))

        mode_lay.addWidget(self.btn_add)
        mode_lay.addWidget(self.btn_sel)
        lay.addWidget(mode_box)

        # Info de superficie y volumen
        info_box = QGroupBox(" Room Info")
        info_lay = QGridLayout(info_box)
        info_lay.addWidget(QLabel("Floor area:"), 0, 0)
        self.area_lbl = QLabel("— m²")
        info_lay.addWidget(self.area_lbl, 0, 1)
        info_lay.addWidget(QLabel("Volume:"), 1, 0)
        self.vol_lbl = QLabel("— m³")
        info_lay.addWidget(self.vol_lbl, 1, 1)
        lay.addWidget(info_box)

        # Altura
        h_box = QGroupBox(" Room Height [m]")
        h_lay = QHBoxLayout(h_box)
        self.height_edit = QLineEdit("3.0")
        self.height_edit.setValidator(QDoubleValidator(0.1, 100, 2, self))
        self.height_edit.textChanged.connect(self._update_room_info)
        h_lay.addWidget(self.height_edit)
        lay.addWidget(h_box)

        # Tabla de vértices
        vt_box = QGroupBox(" Vertices")
        vt_lay = QVBoxLayout(vt_box)
        self.vtable = VertexTable()
        self.vtable.verticesEdited.connect(self._on_table_edit)
        vt_lay.addWidget(self.vtable)
        lay.addWidget(vt_box)

        # Botones
        btn_frame = QFrame()
        btn_grid  = QGridLayout(btn_frame)
        btn_grid.setSpacing(6)

        actions = [
            ("Undo",             "secondary", self.canvas.undo,          0, 0, 1),
            ("Redo",             "secondary", self.canvas.redo,          0, 1, 1),
            ("Clear",            "danger",    self.canvas.clear,         1, 0, 1),
            ("Preview 3D",       "secondary", self._preview_3d,          1, 1, 1),
            ("New Vertex",       "secondary", self._insert_vertex,       2, 0, 1),
            ("Load Room",        "secondary", self._load_room_file,      2, 1, 1),
            ("Save Room",        "success",   self._save_room_file,      3, 0, 1),
            ("To Room Optimize", "success",   self._to_room_optimize,    3, 1, 1),
        ]
        for text, role, cb, r, c, cs in actions:
            btn = QPushButton(text)
            btn.setProperty("role", role)
            btn.clicked.connect(cb)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn_grid.addWidget(btn, r, c, 1, cs)

        btn_grid.setColumnStretch(0, 1)
        btn_grid.setColumnStretch(1, 1)
        lay.addWidget(btn_frame)
        lay.addStretch()

        return panel

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _insert_vertex(self):
        n = len(self.canvas.vertices)
        dlg = InsertVertexDialog(n, self)
        if dlg.exec() == QDialog.Accepted:
            idx = dlg.get_index()
            self.canvas.insert_vertex(idx)

    def _set_mode(self, mode: str):
        self.canvas.set_mode(mode)
        self.btn_add.setChecked(mode == "add")
        self.btn_sel.setChecked(mode == "select")

    def _on_verts_changed(self):
        self.vtable.load(self.canvas.vertices)
        self._update_room_info()

    def _update_room_info(self):
        from geometry import signed_area
        verts = self.canvas.vertices
        if len(verts) < 3:
            self.area_lbl.setText("— m²")
            self.vol_lbl.setText("— m³")
            return
        try:
            area = abs(signed_area(verts))
            height = float(self.height_edit.text() or 0)
            vol = area * height
            self.area_lbl.setText(f"{area:.3f} m²")
            self.vol_lbl.setText(f"{vol:.3f} m³")
        except ValueError:
            self.area_lbl.setText("— m²")
            self.vol_lbl.setText("— m³")

    def _on_table_edit(self, verts: list):
        self.canvas.set_vertices(verts)

    def _on_wall_clicked(self, idx: int):
        wall = self.canvas.wall_props[idx]
        dlg  = WallConfigDialog(wall, self)
        if dlg.exec() == QDialog.Accepted:
            self.canvas.wall_props[idx] = dlg.get_wall()
            self.canvas.update_wall_tilt(idx, dlg.get_wall()["tilt_deg"])

    def _preview_3d(self):
        verts = self.canvas.vertices
        if len(verts) < 3:
            return
        try:
            height = float(self.height_edit.text())
            tilts  = [w["tilt_deg"] for w in self.canvas.wall_props]
            floor, ceiling = compute_ceiling(verts, height, tilts)

            import pyvista as pv
            n  = len(floor)
            pts = np.array([(x,y,0.0) for x,y in floor] +
                           list(ceiling))
            faces = []
            faces.append([n] + list(reversed(range(n))))
            faces.append([n] + list(range(n, 2*n)))
            for i in range(n):
                j = (i+1)%n
                faces.append([4, i, j, j+n, i+n])

            mesh = pv.PolyData(pts, np.hstack(faces))
            pl   = pv.Plotter(off_screen=False)
            pl.set_background("#1e1e1e")
            pl.add_mesh(mesh, show_edges=True, color="silver", opacity=0.5)
            pl.add_axes()
            pl.show()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Preview Error", str(e))

    def _load_room_file(self):
        import json
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Room", "", "JSON Files (*.json)"
        )
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
            self.refresh_from_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")

    def _save_room_file(self):
        import json
        from PySide6.QtWidgets import QFileDialog, QMessageBox
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
            floor, self.canvas.wall_props, height,
            original_verts=verts
        )

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Room", "", "JSON Files (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith('.json'):
            path += '.json'
        with open(path, 'w') as f:
            json.dump({"data": self.state.room_geometry}, f, indent=4)
        print(f"File saved: {path}")

    def _to_room_optimize(self):
        from PySide6.QtWidgets import QMessageBox
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
            floor, self.canvas.wall_props, height,
            original_verts=verts
        )

        # Buscar el tab de Room Optimization y llamar a _load_from_design
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "tabs"):
                opt_tab = parent.tabs.widget(1)
                if hasattr(opt_tab, "_load_from_design"):
                    opt_tab._load_from_design()
                    parent.tabs.setCurrentIndex(1)
                break
            parent = parent.parent()

    def _confirm(self):
        verts = self.canvas.vertices
        if len(verts) < 3:
            return
        try:
            height = float(self.height_edit.text())
            tilts  = [w["tilt_deg"] for w in self.canvas.wall_props]
            floor, ceiling = compute_ceiling(verts, height, tilts)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Geometry Error", str(e))
            return

        geom = build_geometry_dict(floor, self.canvas.wall_props, height)
        self.state.room_geometry = geom

        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Room Confirmed",
            f"Room saved: {len(floor)} walls, height {height} m")