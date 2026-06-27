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
    QTabWidget
)

sys.path.insert(0, os.path.dirname(__file__))
import room_modal_optimizer.gui.dummy_functions as dumF


# ── Estado global ─────────────────────────────────────────────────────────────

class RoomState:
    def __init__(self):
        self.room_geometry: dict = {
            "data": {
                "vertices": {},
                "walls":    {},
                "Z":        3.0
            }
        }
        self.msh_path:    str  = ""
        self.room_name:   str  = "Unnamed Room"
        self.is_symmetric: bool = False

        # ── GA configuration ──────────────────────────────────────
        self.ga_config: dict = {
            "vertex_ranges":  [],   # [{"vertex", "xmin", "xmax", "ymin", "ymax", "enabled"}, ...]
            "wall_ranges":    [],   # [{"wall", "tmin", "tmax", "enabled"}, ...]
            "height_ranges":  {"zmin": 0.0, "zmax": 0.0, "enabled": True},
            "output_dir":     "",
            "target_tr":      0.5,
        }

    def reset(self):
        self.__init__()

    @property
    def symmetric(self) -> bool:
        return self.is_symmetric


# ── Ventana principal ─────────────────────────────────────────────────────────

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Room Modal Optimizer")
        self.resize(1400, 900)

        self.state = RoomState()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.main_page = MainPage(self)
        root.addWidget(self.main_page)

    def go_to_main(self):
        design_tab = self.main_page.tabs.widget(0)
        if hasattr(design_tab, "refresh_from_state"):
            design_tab.refresh_from_state()
        has_data = bool(self.state.room_geometry.get("data", {}).get("vertices"))
        opt_tab = self.main_page.tabs.widget(1)
        if hasattr(opt_tab, "_load_from_design") and has_data:
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
        self.window.go_to_tab(0)

    def _load_room(self):
        from PySide6.QtWidgets import QFileDialog
        import json
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Room Data", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
            self.window.state.room_geometry = raw if "data" in raw else {"data": raw}
            self.window.go_to_main()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{e}")

    def _load_skp(self):
        # Dummy: en el futuro abrirá file dialog .skp
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

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_room_design_tab(),           "Room Design")
        self.tabs.addTab(self._build_room_optimization_tab(),     "Room Optimization")
        self.tabs.addTab(self._build_optimization_results_tab(),  "Optimization Results")
        root.addWidget(self.tabs)

    def load_state(self):
        pass  # room_geometry se gestiona directamente desde los módulos

    # ── Tabs ──────────────────────────────────────────────────────────────────
    def _build_room_design_tab(self) -> QWidget:
        from room_modal_optimizer.gui.room_design import RoomDesignTab
        return RoomDesignTab(self.state)

    def _build_room_optimization_tab(self) -> QWidget:
        from room_modal_optimizer.gui.optim_results import RoomOptimizationTab
        self.opt_tab = RoomOptimizationTab(self.state)

        # Cuando termina la optimización, mostramos los resultados en la
        # tab "Optimization Results" y saltamos automáticamente a ella.
        self.opt_tab.resultsReady.connect(self._on_optim_results_ready)
        return self.opt_tab

    def _build_optimization_results_tab(self) -> QWidget:
        from room_modal_optimizer.gui.optim_results import ResultsScreen
        self.results_tab = ResultsScreen(self.state)

        # Botón "← Back to GA config" dentro de la tab de resultados:
        # nos lleva de vuelta a la tab "Room Optimization".
        self.results_tab.backToConfigRequested.connect(
            lambda: self.tabs.setCurrentIndex(1)
        )
        return self.results_tab

    def _on_optim_results_ready(self, payload):
        bestMsfd, params, bestMicPositions, responses = payload
        self.results_tab.load_results(bestMsfd, params, bestMicPositions, responses)
        self.tabs.setCurrentIndex(2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from room_modal_optimizer.gui.styles import apply_theme
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())