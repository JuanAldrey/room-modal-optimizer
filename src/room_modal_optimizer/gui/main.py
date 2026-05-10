import sys
import os
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QFrame, QLabel, QLineEdit,
    QPushButton, QGridLayout, QHBoxLayout, QVBoxLayout,
    QSizePolicy, QGroupBox, QComboBox
)

sys.path.insert(0, os.path.dirname(__file__))
import dummy_functions as dumF


class RoomModalOptimizer(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Room Modal Optimizer")
        self.resize(1400, 900)
        self._init_vars()

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addWidget(self._build_left_panel(),  stretch=3)
        root.addWidget(self._build_right_panel(), stretch=1)

    def _init_vars(self):
        keys = [
            "Lx", "Ly", "Lz",
            "left_y0", "left_y1", "right_y0", "right_y1",
            "front_x0", "front_x1", "back_x0", "back_x1",
            "left_angle", "right_angle", "front_angle", "back_angle",
        ]
        self.room_params: dict[str, QLineEdit] = {}
        self.msh_path = ""

    # ── Panel izquierdo (mesh + plots) ────────────────────────────────────────
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        lay.addWidget(self._build_mesh_view(),  stretch=3)
        lay.addWidget(self._build_placeholder("Modes Distribution"), stretch=1)
        lay.addWidget(self._build_placeholder("Modal Response"),     stretch=1)

        return panel

    def _build_mesh_view(self) -> QGroupBox:
        box = QGroupBox(" Room Mesh View")
        outer = QVBoxLayout(box)

        # Contenedor con posicionamiento absoluto para superponer el combo
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(container)

        bg = self.palette().color(self.backgroundRole()).name()
        self.fig = plt.Figure(facecolor=bg)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(container)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Combo flotante en esquina superior derecha
        self.view_combo = QComboBox(container)
        for label, key in [("Isometric","iso"), ("Top","xy"), ("Front","xz"), ("Side","yz")]:
            self.view_combo.addItem(label, key)
        self.view_combo.setFixedWidth(110)
        self.view_combo.activated.connect(
            lambda: self._set_view(self.view_combo.currentData())
        )

        # Layout para posicionar canvas y reubicar combo al resize
        canvas_lay = QVBoxLayout(container)
        canvas_lay.setContentsMargins(0, 0, 0, 0)
        canvas_lay.addWidget(self.canvas)

        # Posicionar el combo encima vía evento de resize
        container.resizeEvent = lambda e: self.view_combo.move(
            container.width() - self.view_combo.width() - 8, 8
        )

        return box

    def _build_placeholder(self, title: str) -> QGroupBox:
        box = QGroupBox(f" {title}")
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        box.setMinimumHeight(150)
        lay = QVBoxLayout(box)
        lbl = QLabel(f"[ {title} ]")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #555555; font-size: 14pt;")
        lay.addWidget(lbl)
        return box

    # ── Panel derecho (entries + botones) ─────────────────────────────────────
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        lay.addWidget(self._create_input_group(
            "Plant Lengths [m]:",
            [("Lx","Lx:"), ("Ly","Ly:"), ("Lz","Lz:")],
            cols=3
        ))
        lay.addWidget(self._create_input_group(
            "Plant Offsets [m]:",
            [("left_y0","L-Y0:"), ("left_y1","L-Y1:"),
             ("right_y0","R-Y0:"), ("right_y1","R-Y1:"),
             ("front_x0","F-X0:"), ("front_x1","F-X1:"),
             ("back_x0","B-X0:"),  ("back_x1","B-X1:")],
            cols=4, show_reset=True
        ))
        lay.addWidget(self._create_input_group(
            "Wall Inclination [deg]:",
            [("left_angle","Left:"), ("right_angle","Right:"),
             ("front_angle","Front:"), ("back_angle","Back:")],
            cols=4, show_reset=True
        ))

        lay.addStretch()
        lay.addWidget(self._build_action_buttons())
        return panel

    # ── Helpers de UI ─────────────────────────────────────────────────────────
    def _make_button(self, text, role, callback) -> QPushButton:
        btn = QPushButton(text)
        btn.setProperty("role", role)
        btn.clicked.connect(callback)
        return btn

    def _create_input_group(self, title, items, cols=3, show_reset=False) -> QFrame:
        frame = QFrame()
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 8, 0, 4)
        outer.setSpacing(4)

        header = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setFont(QFont("Helvetica", 10, QFont.Bold))
        header.addWidget(lbl)
        if show_reset:
            keys = [k for k, _ in items]
            rst = self._make_button("reset to 0", "link", lambda _, ks=keys: self._set_to_zero(ks))
            header.addWidget(rst)
        header.addStretch()
        outer.addLayout(header)

        grid = QGridLayout()
        grid.setSpacing(4)
        for i, (key, label) in enumerate(items):
            r, c = divmod(i, cols)
            grid.addWidget(QLabel(label), r, c * 2, Qt.AlignLeft)
            entry = QLineEdit()
            entry.setFixedWidth(60)
            grid.addWidget(entry, r, c * 2 + 1, Qt.AlignLeft)
            self.room_params[key] = entry
        for c in range(cols):
            grid.setColumnStretch(c * 2 + 1, 1)
        outer.addLayout(grid)
        return frame

    def _build_action_buttons(self) -> QFrame:
        frame = QFrame()
        grid  = QGridLayout(frame)
        grid.setContentsMargins(0, 10, 0, 10)
        grid.setSpacing(6)

        specs = [
            ("Calculate",            "success",   self.execute_pipeline, 0, 0),
            ("Export Room Data",     "secondary",  self.export_rd,        0, 1),
            ("Export Modal Response","secondary",  self.export_mr,        1, 0),
            ("Clear",                "danger",     self.clear_entries,    1, 1),
        ]
        for text, role, cmd, r, c in specs:
            btn = self._make_button(text, role, cmd)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            grid.addWidget(btn, r, c)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return frame

    # ── Renderizado ───────────────────────────────────────────────────────────
    def display_mesh_pyvista(self, msh_path: str, view: str = "iso") -> None:
        if not msh_path:
            return
        try:
            bg      = self.palette().color(self.backgroundRole()).name()
            mesh    = pv.read(msh_path)
            surface = mesh.extract_surface(algorithm='dataset_surface')

            plotter = pv.Plotter(off_screen=True)
            plotter.set_background(bg)
            plotter.add_mesh(surface, color="silver", show_edges=False, opacity=0.3)

            if   view == "xy":  plotter.view_xy()
            elif view == "xz":  plotter.view_xz()
            elif view == "yz":  plotter.view_yz()
            else:               plotter.view_isometric()

            plotter.reset_camera()
            screenshot = plotter.screenshot()
            plotter.close()

            self.fig.clear()
            self.fig.set_facecolor(bg)
            ax = self.fig.add_subplot(111)
            ax.set_facecolor(bg)
            ax.imshow(screenshot)
            ax.axis("off")
            self.fig.tight_layout(pad=0)
            self.canvas.draw()

        except Exception as e:
            print(f"Error PyVista Render: {e}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _set_view(self, view: str) -> None:
        if self.msh_path:
            self.display_mesh_pyvista(self.msh_path, view)

    def _set_to_zero(self, keys):
        for k in keys:
            self.room_params[k].setText("0")

    def execute_pipeline(self):
        try:
            data = {k: float(w.text()) for k, w in self.room_params.items()}
        except ValueError:
            print("Error: todos los campos deben ser numéricos.")
            return
        path = dumF.gui_get_mesh_path(data)
        if path:
            self.msh_path = path
            self.display_mesh_pyvista(path)

    def export_rd(self):  print("Exporting Room Data...")
    def export_mr(self):  print("Exporting Modal Response...")

    def clear_entries(self):
        for w in self.room_params.values():
            w.clear()
        self.fig.clear()
        self.canvas.draw()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from styles import apply_theme
    app = QApplication(sys.argv)
    apply_theme(app)
    win = RoomModalOptimizer()
    win.show()
    sys.exit(app.exec())