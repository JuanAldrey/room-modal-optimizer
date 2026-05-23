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
    QSizePolicy, QGroupBox, QComboBox, QMessageBox,
    QStackedWidget, QSpacerItem, QTabWidget
)

sys.path.insert(0, os.path.dirname(__file__))
import dummy_functions as dumF


# ── Estado global del recinto ─────────────────────────────────────────────────
class RoomState:
    def __init__(self):
        # {"V1":(x,y), ..., "Vn":(x,y), "W1":ang, ..., "Wn":ang, "Z":h}
        self.room_geometry: dict = {}
        self.source_pos: tuple = (0.0, 0.0, 0.0)
        self.mic_pos:    tuple = (0.0, 0.0, 0.0)
        self.msh_path:   str   = ""

    def reset(self):
        self.__init__()


# ── Ventana principal ─────────────────────────────────────────────────────────
class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Room Modal Optimizer")
        self.resize(1400, 900)

        self.state = RoomState()

        self.stack = QStackedWidget(self)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.stack)

        self.welcome_page = WelcomePage(self)
        self.main_page    = MainPage(self)

        self.stack.addWidget(self.welcome_page)
        self.stack.addWidget(self.main_page)
        self.stack.setCurrentWidget(self.welcome_page)

    def go_to_main(self):
        self.main_page.load_state()
        self.stack.setCurrentWidget(self.main_page)
        design_tab = self.main_page.tabs.widget(0)
        if hasattr(design_tab, "refresh_from_state"):
            design_tab.refresh_from_state()
        opt_tab = self.main_page.tabs.widget(1)
        if hasattr(opt_tab, "_load_from_design") and self.state.room_geometry:
            opt_tab._load_from_design()

    def go_to_tab(self, index: int):
        self.main_page.tabs.setCurrentIndex(index)


# ── Página de bienvenida ──────────────────────────────────────────────────────
class WelcomePage(QWidget):

    def __init__(self, window: MainWindow):
        super().__init__(window)
        self.window = window
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(20)

        title = QLabel("Welcome to\nRoom Modal Optimizer")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        lay.addWidget(title)

        lay.addSpacerItem(QSpacerItem(0, 30, QSizePolicy.Minimum, QSizePolicy.Fixed))

        btn_frame = QFrame()
        btn_lay   = QHBoxLayout(btn_frame)
        btn_lay.setSpacing(16)
        btn_lay.setAlignment(Qt.AlignCenter)

        for text, role, cb in [
            ("Create Room", "success",   self._create_room),
            ("Load Room",   "secondary", self._load_room),
            ("Load SKP",    "secondary", self._load_skp),
        ]:
            btn = QPushButton(text)
            btn.setProperty("role", role)
            btn.setFixedSize(160, 48)
            btn.clicked.connect(cb)
            btn_lay.addWidget(btn)

        lay.addWidget(btn_frame)

    def _create_room(self):
        self.window.state.reset()
        self.window.go_to_main()
        self.window.go_to_tab(0)  # Room Design

    def _load_room(self):
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import json
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Room Data", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
            geom = raw.get("data", raw)
            # Convertir listas a listas (ya es el formato correcto)
            self.window.state.room_geometry = geom
            self.window.go_to_main()
            design_tab = self.window.main_page.tabs.widget(0)
            if hasattr(design_tab, "refresh_from_state"):
                design_tab.refresh_from_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{e}")

    def _load_skp(self):
        self.window.state.reset()
        self.window.go_to_main()


# ── Página principal ──────────────────────────────────────────────────────────
class MainPage(QWidget):

    def __init__(self, window: MainWindow):
        super().__init__(window)
        self.window = window
        self._build()

    @property
    def state(self) -> RoomState:
        return self.window.state

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Inicializar dicts antes de construir paneles
        self.source_entries: dict[str, QLineEdit] = {}
        self.mic_entries:    dict[str, QLineEdit] = {}

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_room_design_tab(),           "Room Design")
        self.tabs.addTab(self._build_room_optimization_tab(),     "Room Optimization")
        self.tabs.addTab(self._build_transfer_simulation_tab(),   "Transfer Simulation")
        root.addWidget(self.tabs)

    def load_state(self):
        src = self.state.source_pos
        mic = self.state.mic_pos
        for k, w in self.source_entries.items():
            w.setText(str(src[["src_x","src_y","src_z"].index(k)]))
        for k, w in self.mic_entries.items():
            w.setText(str(mic[["mic_x","mic_y","mic_z"].index(k)]))

    def _sync_state(self):
        src = self.source_entries
        mic = self.mic_entries
        self.state.source_pos = (
            float(src["src_x"].text() or 0),
            float(src["src_y"].text() or 0),
            float(src["src_z"].text() or 0),
        )
        self.state.mic_pos = (
            float(mic["mic_x"].text() or 0),
            float(mic["mic_y"].text() or 0),
            float(mic["mic_z"].text() or 0),
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    def _build_room_design_tab(self) -> QWidget:
        from room_design import RoomDesignTab
        return RoomDesignTab(self.state)

    def _build_room_optimization_tab(self) -> QWidget:
        from room_optimization import RoomOptimizationTab
        return RoomOptimizationTab(self.state)

    def _build_transfer_simulation_tab(self) -> QWidget:
        tab = QWidget()
        lbl = QLabel("[ Transfer Simulation ]")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #555555; font-size: 14pt;")
        lay = QVBoxLayout(tab)
        lay.addWidget(lbl)
        return tab

    # ── Panel izquierdo ───────────────────────────────────────────────────────
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(self._build_mesh_view(),                       stretch=3)
        lay.addWidget(self._build_placeholder("Modes Distribution"), stretch=1)
        lay.addWidget(self._build_placeholder("Modal Response"),     stretch=1)
        return panel

    def _build_mesh_view(self) -> QGroupBox:
        box   = QGroupBox(" Room Mesh View")
        outer = QVBoxLayout(box)

        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(container)

        self.fig = plt.Figure(facecolor="#000000")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #000000;")
        self.canvas.setParent(container)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.view_combo = QComboBox(container)
        for label, key in [("Isometric","iso"),("Top","xy"),("Front","xz"),("Side","yz")]:
            self.view_combo.addItem(label, key)
        self.view_combo.setFixedWidth(110)
        self.view_combo.activated.connect(
            lambda: self._set_view(self.view_combo.currentData())
        )

        canvas_lay = QVBoxLayout(container)
        canvas_lay.setContentsMargins(0, 0, 0, 0)
        canvas_lay.addWidget(self.canvas)

        self.overlay = QLabel("Calculating...", container)
        self.overlay.setAlignment(Qt.AlignCenter)
        self.overlay.setStyleSheet("""
            background-color: rgba(0,0,0,160);
            color: white; font-size: 16pt; font-weight: bold; border-radius: 8px;
        """)
        self.overlay.hide()

        container.resizeEvent = lambda e: (
            self.view_combo.move(container.width() - self.view_combo.width() - 8, 8),
            self.overlay.setGeometry(0, 0, container.width(), container.height())
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

    # ── Panel derecho ─────────────────────────────────────────────────────────
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        lay.addWidget(self._input_group(
            "Source Position [m]:",
            [("src_x","X:"),("src_y","Y:"),("src_z","Z:")],
            cols=3, d=self.source_entries
        ))
        lay.addWidget(self._input_group(
            "Microphone Position [m]:",
            [("mic_x","X:"),("mic_y","Y:"),("mic_z","Z:")],
            cols=3, d=self.mic_entries
        ))

        lay.addStretch()
        lay.addWidget(self._build_action_buttons())
        return panel

    # ── Helpers UI ────────────────────────────────────────────────────────────
    def _btn(self, text, role, cb) -> QPushButton:
        btn = QPushButton(text)
        btn.setProperty("role", role)
        btn.clicked.connect(cb)
        return btn

    def _input_group(self, title, items, cols=3, show_reset=False, d=None) -> QFrame:
        if d is None: d = self.room_entries
        frame = QFrame()
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 8, 0, 4)
        outer.setSpacing(4)

        header = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        header.addWidget(lbl)
        if show_reset:
            keys = [k for k, _ in items]
            header.addWidget(self._btn("reset to 0", "link",
                lambda _, ks=keys, dd=d: [dd[k].setText("0") for k in ks]))
        header.addStretch()
        outer.addLayout(header)

        grid = QGridLayout()
        grid.setSpacing(4)
        for i, (key, label) in enumerate(items):
            r, c = divmod(i, cols)
            grid.addWidget(QLabel(label), r, c*2, Qt.AlignLeft)
            entry = QLineEdit()
            entry.setFixedWidth(60)
            grid.addWidget(entry, r, c*2+1, Qt.AlignLeft)
            d[key] = entry
        for c in range(cols):
            grid.setColumnStretch(c*2+1, 1)
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
            btn = self._btn(text, role, cmd)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            grid.addWidget(btn, r, c)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return frame

    # ── Renderizado ───────────────────────────────────────────────────────────
    def display_mesh_pyvista(self, msh_path: str, view: str = "iso") -> None:
        if not msh_path: return
        try:
            mesh    = pv.read(msh_path)
            surface = mesh.extract_surface(algorithm='dataset_surface')
            plotter = pv.Plotter(off_screen=True)
            plotter.set_background("#000000")
            plotter.add_mesh(surface, color="silver", show_edges=False, opacity=0.3)
            if   view == "xy": plotter.view_xy()
            elif view == "xz": plotter.view_xz()
            elif view == "yz": plotter.view_yz()
            else:              plotter.view_isometric()
            plotter.reset_camera()
            screenshot = plotter.screenshot()
            plotter.close()
            self.fig.clear()
            self.fig.set_facecolor("#000000")
            ax = self.fig.add_subplot(111)
            ax.set_facecolor("#000000")
            ax.imshow(screenshot)
            ax.axis("off")
            self.fig.tight_layout(pad=0)
            self.canvas.draw()
        except Exception as e:
            self._show_error(f"Error al renderizar el mesh:\n{e}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _set_view(self, view: str):
        if self.state.msh_path:
            self.display_mesh_pyvista(self.state.msh_path, view)

    def _show_error(self, msg: str):
        dlg = QMessageBox(self)
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

    def execute_pipeline(self):
        try:
            self._sync_state()
        except ValueError:
            self._show_error("Todos los campos deben ser numéricos.")
            return
        self.overlay.show()
        self.overlay.raise_()
        QApplication.processEvents()
        try:
            data = dict(self.state.room_geometry)
            data["source_pos"] = self.state.source_pos
            data["mic_pos"]    = self.state.mic_pos
            path = dumF.gui_get_mesh_path(data)
        except Exception as e:
            self.overlay.hide()
            self._show_error(f"Error al generar el mesh:\n{e}")
            return
        if path:
            self.state.msh_path = path
            self.display_mesh_pyvista(path)
        self.overlay.hide()

    def export_rd(self):  print("Exporting Room Data...")
    def export_mr(self):  print("Exporting Modal Response...")

    def clear_entries(self):
        for w in list(self.source_entries.values()) + \
                 list(self.mic_entries.values()):
            w.clear()
        self.state.reset()
        self.fig.clear()
        self.canvas.draw()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from styles import apply_theme
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())