"""
room_design.py  –  Floor-plan editor for the Room Modal Optimizer.

Classes
-------
WallConfigDialog      – Dialog to set wall inclination.
InsertVertexDialog    – Dialog to insert a vertex at a chosen position.
CancelSymmetryDialog  – Dialog shown before cancelling symmetric mode.
FloorCanvas           – Interactive matplotlib canvas: Add / Select / Delete
                        / Place / Audience / Source modes, scroll-zoom,
                        middle-drag pan, undo-redo, symmetric drawing (X = 0).
SourceTable           – Editable QTableWidget for multiple sources (#, X, Y, Z).
VertexTable           – Editable QTableWidget mirroring a vertex list.
RoomDesignTab         – Full tab widget wiring canvas + tables + controls.

Symmetric rooms
---------------
When "Symmetric Room" is active every click in "Add Vertices" / "Add Audience"
mode is mirrored across X = 0 automatically.  Sources are pinned to X = 0.
The full polygon (both halves) is always written to JSON; on load the X ≥ 0
half is re-derived in memory to restore the live mirrored-editing chain.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pyvista as pv

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from room_modal_optimizer.gui.geometry import (
    build_geometry_dict, clip_polygon_halfplane, compute_ceiling,
    nearest_wall, point_side, project_point_on_line, reflect_point,
    room_volume, signed_area,
)

# ── Constants ──────────────────────────────────────────────────────────────────

_SYM_AXIS   = ((0.0, 0.0), (0.0, 1.0))   # fixed symmetry axis: X = 0
_ZOOM_STEP  = 1.2
_ZOOM_MIN   = 0.05
_ZOOM_MAX   = 500.0
_SNAP_FRAC  = 0.015                        # axis-snap as fraction of view diagonal
_VTOL       = 0.35                         # vertex pick tolerance [data units]
_PAN_BUTTON = 2                            # middle mouse button
_AXIS_TOL   = 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# Dialogs
# ═══════════════════════════════════════════════════════════════════════════════

class WallConfigDialog(QDialog):
    """Set inclination angle for a single wall."""

    def __init__(self, wall: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Configure {wall['id']}")
        self.setMinimumWidth(280)
        self.wall = dict(wall)

        form = QFormLayout(self)
        self._tilt = QLineEdit(str(self.wall["tilt_deg"]))
        self._tilt.setValidator(QDoubleValidator(-89.0, 89.0, 2, self))
        form.addRow("Inclination [deg]:", self._tilt)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _accept(self):
        try:
            self.wall["tilt_deg"] = float(self._tilt.text())
        except ValueError:
            return
        self.accept()

    def get_wall(self) -> dict:
        return self.wall


class InsertVertexDialog(QDialog):
    """Choose where to insert a new vertex in the polygon."""

    def __init__(self, n: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Insert New Vertex")
        self.setMinimumWidth(240)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            f"Current vertices: 1 – {n}\n"
            f"Insert position (1 = before V1,  {n+1} = after V{n}):"
        ))
        self._edit = QLineEdit("1")
        self._edit.setValidator(QIntValidator(1, n + 1, self))
        lay.addWidget(self._edit)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_index(self) -> int:
        return int(self._edit.text()) - 1


class CancelSymmetryDialog(QDialog):
    """Warn the user before cancelling symmetric mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cancel Symmetry")
        self.setMinimumWidth(380)
        lay = QVBoxLayout(self)
        lay.setSpacing(12)

        lbl = QLabel(
            "If you cancel symmetry, the current floor plan will be kept.\n\n"
            "However, the room will no longer be optimized maintaining symmetry "
            "— both halves will be treated as independent walls."
        )
        lbl.setWordWrap(True)
        lay.addWidget(lbl)

        self._chk = QCheckBox("Don't show this message again")
        lay.addWidget(self._chk)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Cancel Symmetry")
        btns.button(QDialogButtonBox.Cancel).setText("Keep Symmetry")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def dont_show_again(self) -> bool:
        return self._chk.isChecked()


# ═══════════════════════════════════════════════════════════════════════════════
# Floor canvas
# ═══════════════════════════════════════════════════════════════════════════════

class FloorCanvas(FigureCanvas):
    """
    Matplotlib canvas with interactive floor-plan editing.

    Modes
    -----
    add             – append a floor vertex (mirrored if symmetric mode active).
    select          – click a wall to configure it (fires wallClicked).
    delete          – click to remove the nearest floor vertex (min 3 kept).
    place           – click to position a freshly inserted vertex.
    add_audience    – append a vertex to the audience-area polygon (mirrored).
    delete_audience – click to remove the nearest audience-area vertex.
    place_source    – click to add a source at (X, Y); X pinned to 0 if symmetric.
    delete_source   – click to remove the nearest source.

    Signals
    -------
    verticesChanged   – any floor-vertex mutation.
    wallClicked(int)  – wall selected in "select" mode.
    symModeChanged(bool)  – symmetric mode toggled.
    symStatusChanged(str) – human-readable status string for symmetric mode.
    audienceChanged   – any audience-area mutation.
    sourceChanged     – any source list mutation.
    """

    verticesChanged  = Signal()
    wallClicked      = Signal(int)
    symModeChanged   = Signal(bool)
    symStatusChanged = Signal(str)
    audienceChanged  = Signal()
    sourceChanged    = Signal()

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, parent=None):
        fig = Figure(facecolor="#1e1e1e")
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax = fig.add_subplot(111)
        self._style_ax()

        # Core data
        self.vertices:          list[tuple[float, float]] = []
        self.wall_props:        list[dict]                = []
        self.audience_vertices: list[tuple[float, float]] = []
        self.sources:           list[tuple[float, float]] = []  # (x, y); Z in SourceTable

        # Undo / redo
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []

        # Interaction
        self._mode:      str      = "add"
        self._place_idx: int|None = None
        self._sel_wall:  int|None = None

        # Zoom
        self._autofit:  bool       = True
        self._cur_xlim: tuple|None = None
        self._cur_ylim: tuple|None = None

        # Pan
        self._panning:      bool       = False
        self._pan_start_px: tuple|None = None
        self._pan_xlim0:    tuple|None = None
        self._pan_ylim0:    tuple|None = None

        # Symmetric-room state (floor)
        self._sym_active: bool        = False
        self._sym_chain:  list[tuple] = []
        self._sym_side:   int|None    = None   # +1 / -1 once locked

        # Symmetric audience state (independent chain/side, same axis)
        self._aud_sym_chain: list[tuple] = []
        self._aud_sym_side:  int|None    = None

        # Symmetric source state: side lock for mirrored source placement
        self._src_sym_side: int|None = None

        self.mpl_connect("button_press_event",   self._on_click)
        self.mpl_connect("scroll_event",         self._on_scroll)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    # ── Axis style ────────────────────────────────────────────────────────────

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

    # ── Undo / redo ───────────────────────────────────────────────────────────

    def _snapshot(self) -> dict:
        return {
            "vertices":          list(self.vertices),
            "mode":              self._mode,
            "sym_active":        self._sym_active,
            "sym_chain":         list(self._sym_chain),
            "sym_side":          self._sym_side,
            "audience_vertices": list(self.audience_vertices),
            "source_pos":        list(self.sources),
            "aud_sym_chain":     list(self._aud_sym_chain),
            "aud_sym_side":      self._aud_sym_side,
            "src_sym_side":      self._src_sym_side,
        }

    def _apply_snapshot(self, snap: dict):
        self.vertices          = list(snap["vertices"])
        self._mode             = snap["mode"]
        self._sym_active       = snap["sym_active"]
        self._sym_chain        = list(snap["sym_chain"])
        self._sym_side         = snap["sym_side"]
        self.audience_vertices = list(snap.get("audience_vertices", []))
        self.sources           = list(snap.get("source_pos", []))
        self._aud_sym_chain    = list(snap.get("aud_sym_chain", []))
        self._aud_sym_side     = snap.get("aud_sym_side")
        self._src_sym_side     = snap.get("src_sym_side")
        self._place_idx        = None
        self._rebuild_wall_props()
        self._redraw()
        self.verticesChanged.emit()
        self.audienceChanged.emit()
        self.sourceChanged.emit()
        self.symModeChanged.emit(self._sym_active)
        self.symStatusChanged.emit(self._status_text())

    def _save_undo(self):
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()

    def undo(self):
        if self._undo_stack:
            self._redo_stack.append(self._snapshot())
            self._apply_snapshot(self._undo_stack.pop())

    def redo(self):
        if self._redo_stack:
            self._undo_stack.append(self._snapshot())
            self._apply_snapshot(self._redo_stack.pop())

    # ── Public mutations (floor) ──────────────────────────────────────────────

    def set_mode(self, mode: str):
        self._mode      = mode
        self._place_idx = None
        self._sel_wall  = None
        self._redraw()

    def set_vertices(self, verts: list[tuple[float, float]]):
        """Replace vertices (from table edits), propagating mirror when active."""
        self._save_undo()
        if self._sym_active and self._sym_chain:
            changed_idx = next(
                (i for i, (o, n) in enumerate(zip(self.vertices, verts))
                 if abs(o[0]-n[0]) > 1e-9 or abs(o[1]-n[1]) > 1e-9),
                None,
            )
            if changed_idx is not None:
                m = len(self._sym_chain)
                a, b = _SYM_AXIS
                if changed_idx < m:
                    self._sym_chain[changed_idx] = verts[changed_idx]
                else:
                    new_chain_pt = reflect_point(verts[changed_idx], a, b)
                    ci = self._vertex_mirror_map().get(changed_idx)
                    if ci is not None and ci < m:
                        self._sym_chain[ci] = new_chain_pt
                self._rebuild_sym_vertices()
        else:
            self.vertices = list(verts)
        self._rebuild_wall_props()
        self._redraw()
        self.verticesChanged.emit()

    def insert_vertex(self, idx: int):
        """Insert a placeholder vertex and enter 'place' mode."""
        self._save_undo()
        self.vertices.insert(idx, (0.0, 0.0))
        self._rebuild_wall_props()
        self._place_idx = idx
        self._mode      = "place"
        self._redraw()
        self.verticesChanged.emit()

    def delete_vertex(self, idx: int):
        """Remove vertex at idx; in symmetric mode also removes its mirror."""
        if len(self.vertices) <= 3:
            return
        self._save_undo()
        if self._sym_active and self._sym_chain:
            m         = len(self._sym_chain)
            chain_idx = idx if idx < m else self._vertex_mirror_map().get(idx)
            if chain_idx is not None and 0 <= chain_idx < m and len(self._sym_chain) > 1:
                self._sym_chain.pop(chain_idx)
                self._rebuild_sym_vertices()
            else:
                self.vertices.pop(idx)
        else:
            self.vertices.pop(idx)
        self._rebuild_wall_props()
        self._redraw()
        self.verticesChanged.emit()

    def update_wall_tilt(self, idx: int, tilt: float):
        """Set tilt for wall idx; propagates to mirror wall when symmetric."""
        if not (0 <= idx < len(self.wall_props)):
            return
        self.wall_props[idx]["tilt_deg"] = tilt
        if self._sym_active:
            mirror_idx = self._wall_mirror_map().get(idx)
            if mirror_idx is not None and mirror_idx != idx:
                self.wall_props[mirror_idx]["tilt_deg"] = tilt
        self._redraw()

    # ── Public mutations (audience) ───────────────────────────────────────────

    def set_audience_vertices(self, verts: list[tuple[float, float]]):
        """Replace audience polygon (from table edits), propagating mirror."""
        self._save_undo()
        if self._sym_active and self._aud_sym_chain:
            changed_idx = next(
                (i for i, (o, n) in enumerate(zip(self.audience_vertices, verts))
                 if abs(o[0]-n[0]) > 1e-9 or abs(o[1]-n[1]) > 1e-9),
                None,
            )
            if changed_idx is not None:
                m = len(self._aud_sym_chain)
                a, b = _SYM_AXIS
                if changed_idx < m:
                    self._aud_sym_chain[changed_idx] = verts[changed_idx]
                else:
                    new_chain_pt = reflect_point(verts[changed_idx], a, b)
                    ci = self._aud_vertex_mirror_map().get(changed_idx)
                    if ci is not None and ci < m:
                        self._aud_sym_chain[ci] = new_chain_pt
                self._rebuild_aud_sym_vertices()
            elif len(verts) != len(self.audience_vertices):
                self.audience_vertices = list(verts)
                self._aud_sym_chain    = []
                self._aud_sym_side     = None
        else:
            self.audience_vertices = list(verts)
        self._redraw()
        self.audienceChanged.emit()

    def add_audience_vertex(self, x: float, y: float):
        self._save_undo()
        self.audience_vertices.append((round(x, 3), round(y, 3)))
        self._redraw()
        self.audienceChanged.emit()

    def delete_audience_vertex(self, idx: int):
        if not (0 <= idx < len(self.audience_vertices)):
            return
        self._save_undo()
        if self._sym_active and self._aud_sym_chain:
            m         = len(self._aud_sym_chain)
            chain_idx = idx if idx < m else self._aud_vertex_mirror_map().get(idx)
            if chain_idx is not None and 0 <= chain_idx < m:
                self._aud_sym_chain.pop(chain_idx)
                self._rebuild_aud_sym_vertices()
            else:
                self.audience_vertices.pop(idx)
        else:
            self.audience_vertices.pop(idx)
        self._redraw()
        self.audienceChanged.emit()

    def clear_audience(self):
        self._save_undo()
        self.audience_vertices.clear()
        self._aud_sym_chain.clear()
        self._aud_sym_side = None
        self._redraw()
        self.audienceChanged.emit()

    # ── Public mutations (sources) ────────────────────────────────────────────

    def add_source(self, x: float, y: float):
        """Add a source at (x, y); in symmetric mode also adds the mirror."""
        self._save_undo()
        p = (round(x, 3), round(y, 3))
        self.sources.append(p)
        if self._sym_active:
            a, b = _SYM_AXIS
            mirror = reflect_point(p, a, b)
            # Only add mirror when point is not on the axis
            if abs(mirror[0] - p[0]) > 1e-6 or abs(mirror[1] - p[1]) > 1e-6:
                self.sources.append((round(mirror[0], 3), round(mirror[1], 3)))
        self._redraw()
        self.sourceChanged.emit()

    def delete_source(self, idx: int):
        if not (0 <= idx < len(self.sources)):
            return
        self._save_undo()
        self.sources.pop(idx)
        self._redraw()
        self.sourceChanged.emit()

    def clear_sources(self):
        self._save_undo()
        self.sources.clear()
        self._redraw()
        self.sourceChanged.emit()

    # ── General mutations ─────────────────────────────────────────────────────

    def clear(self):
        self._save_undo()
        self.vertices.clear()
        self.wall_props.clear()
        self.audience_vertices.clear()
        self.sources.clear()
        self._sel_wall  = None
        self._place_idx = None
        self._mode      = "add"
        self._autofit   = True
        self._cur_xlim  = None
        self._cur_ylim  = None
        self._exit_sym()
        self._redraw()
        self.verticesChanged.emit()
        self.audienceChanged.emit()
        self.sourceChanged.emit()
        self.symModeChanged.emit(False)
        self.symStatusChanged.emit("")

    def reset_zoom(self):
        self._autofit  = True
        self._cur_xlim = None
        self._cur_ylim = None
        self._redraw()

    def load_geometry(
        self,
        vertices:          list[tuple[float, float]],
        wall_props:        list[dict],
        is_symmetric:      bool,
        audience_vertices: list[tuple[float, float]] | None = None,
        sources:           list[tuple[float, float]] | None = None,
    ):
        """
        Replace canvas content from saved/loaded data.

        `vertices`/`wall_props` always carry the FULL polygon. When
        `is_symmetric` is True the X ≥ 0 half is re-derived in memory to
        restore the live mirrored-editing chain.
        """
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._sel_wall         = None
        self._place_idx        = None
        self._mode             = "add"
        self.audience_vertices = list(audience_vertices) if audience_vertices else []
        self.sources           = [tuple(s) for s in sources] if sources else []

        if is_symmetric and vertices:
            pos_idx  = _positive_x_indices(vertices)
            chain    = [vertices[i] for i in pos_idx]
            a, b     = _SYM_AXIS
            mirrored = [reflect_point(p, a, b) for p in reversed(chain)]
            self.vertices    = _dedupe(chain + mirrored)
            self._sym_active = True
            self._sym_chain  = chain
            self._sym_side   = None

            if self.audience_vertices:
                aud_pos = _positive_x_indices(self.audience_vertices)
                self._aud_sym_chain = [self.audience_vertices[i] for i in aud_pos]
            else:
                self._aud_sym_chain = []
            self._aud_sym_side = None
            self._src_sym_side = None

            self.wall_props = []
            self._rebuild_wall_props()
            m = len(chain)
            for i, src_i in enumerate(pos_idx[:m]):
                if src_i < len(wall_props) and i < len(self.wall_props):
                    self.wall_props[i]["tilt_deg"] = wall_props[src_i].get("tilt_deg", 0.0)
            mirror_map = self._wall_mirror_map()
            for i in range(m):
                j = mirror_map.get(i)
                if j is not None:
                    self.wall_props[j]["tilt_deg"] = self.wall_props[i]["tilt_deg"]
        else:
            self.vertices   = list(vertices)
            self.wall_props = [dict(w) for w in wall_props] if wall_props else []
            self._exit_sym()
            if not self.wall_props:
                self._rebuild_wall_props()

        self._redraw()
        self.verticesChanged.emit()
        self.audienceChanged.emit()
        self.sourceChanged.emit()
        self.symModeChanged.emit(self._sym_active)
        self.symStatusChanged.emit(self._status_text())

    # ── Symmetric mode ────────────────────────────────────────────────────────

    def _exit_sym(self):
        self._sym_active    = False
        self._sym_chain     = []
        self._sym_side      = None
        self._aud_sym_chain = []
        self._aud_sym_side  = None
        self._src_sym_side  = None

    def start_symmetric_room(self):
        """Clear the plan and activate symmetric drawing (axis X = 0)."""
        self._save_undo()
        self.vertices    = []
        self.wall_props  = []
        self._sel_wall   = None
        self._place_idx  = None
        self._mode       = "add"
        self._sym_active = True
        self._sym_chain  = []
        self._sym_side   = None

        self.audience_vertices = []
        self._aud_sym_chain    = []
        self._aud_sym_side     = None
        self._src_sym_side     = None
        self._redraw()
        self.verticesChanged.emit()
        self.audienceChanged.emit()
        self.sourceChanged.emit()
        self.symModeChanged.emit(True)
        self.symStatusChanged.emit(self._status_text())

    def cancel_symmetric_room(self):
        """Exit symmetric mode, keeping whatever polygon was built."""
        self._exit_sym()
        self._mode = "add"
        self._redraw()
        self.symModeChanged.emit(False)
        self.symStatusChanged.emit("")

    # ── Symmetric helpers ─────────────────────────────────────────────────────

    def _rebuild_sym_vertices(self):
        a, b = _SYM_AXIS
        mirrored = [reflect_point(p, a, b) for p in reversed(self._sym_chain)]
        self.vertices = _dedupe(self._sym_chain + mirrored)

    def _rebuild_aud_sym_vertices(self):
        a, b = _SYM_AXIS
        mirrored = [reflect_point(p, a, b) for p in reversed(self._aud_sym_chain)]
        self.audience_vertices = _dedupe(self._aud_sym_chain + mirrored)

    def _vertex_mirror_map(self) -> dict[int, int]:
        """Return {vertex_idx: mirror_vertex_idx} by geometric reflection."""
        return self._mirror_map(self.vertices)

    def _aud_vertex_mirror_map(self) -> dict[int, int]:
        """Return {audience_idx: mirror_audience_idx} by geometric reflection."""
        return self._mirror_map(self.audience_vertices)

    def _mirror_map(self, verts: list[tuple]) -> dict[int, int]:
        if not self._sym_active:
            return {}
        a, b = _SYM_AXIS
        n    = len(verts)
        mm: dict[int, int] = {}
        for i in range(n):
            if i in mm:
                continue
            r      = reflect_point(verts[i], a, b)
            best_j = min(
                (j for j in range(n) if j != i),
                key=lambda j: math.hypot(verts[j][0]-r[0], verts[j][1]-r[1]),
                default=None,
            )
            if best_j is not None:
                d = math.hypot(verts[best_j][0]-r[0], verts[best_j][1]-r[1])
                if d < 1e-4:
                    mm[i]      = best_j
                    mm[best_j] = i
        return mm

    def _wall_mirror_map(self) -> dict[int, int]:
        """Return {wall_idx: mirror_wall_idx} by midpoint reflection."""
        if not self._sym_active or len(self._sym_chain) < 2:
            return {}
        a, b  = _SYM_AXIS
        verts = self.vertices
        n     = len(verts)
        mids  = [
            ((verts[i][0]+verts[(i+1)%n][0])/2,
             (verts[i][1]+verts[(i+1)%n][1])/2)
            for i in range(n)
        ]
        mm: dict[int, int] = {}
        for i in range(n):
            if i in mm:
                continue
            mr     = reflect_point(mids[i], a, b)
            best_j = min(
                (j for j in range(n) if j != i),
                key=lambda j: math.hypot(mids[j][0]-mr[0], mids[j][1]-mr[1]),
                default=None,
            )
            if best_j is not None:
                d = math.hypot(mids[best_j][0]-mr[0], mids[best_j][1]-mr[1])
                if d < 1e-4:
                    mm[i]      = best_j
                    mm[best_j] = i
        return mm

    def _axis_snap_tol(self) -> float:
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        return math.hypot(xl[1]-xl[0], yl[1]-yl[0]) * _SNAP_FRAC

    def _status_text(self) -> str:
        if not self._sym_active:
            return ""
        if self._mode == "add":
            if self._sym_side is None:
                return "Symmetric Room: click vertices on either side to start drawing."
            return ("Symmetric Room: click on the highlighted half — "
                    "the mirror is drawn automatically. "
                    "(Select Wall / Delete Vertex remain available.)")
        if self._mode == "add_audience":
            if self._aud_sym_side is None:
                return "Symmetric Audience: click vertices on either side to start drawing."
            return ("Symmetric Audience: click on the highlighted half — "
                    "the mirror is drawn automatically.")
        if self._mode == "place_source":
            if self._src_sym_side is None:
                return "Symmetric Room: click a source on either side — its mirror will be placed automatically."
            return ("Symmetric Room: click on the highlighted half — "
                    "the mirror source is placed automatically.")
        return ""

    # ── Wall props ────────────────────────────────────────────────────────────

    def _rebuild_wall_props(self):
        old = {w["id"]: w for w in self.wall_props}
        self.wall_props = [
            old.get(f"W{i+1}", {
                "id": f"W{i+1}", "tilt_deg": 0.0,
                "locked": False, "optimize_tilt": False,
                "tilt_min": 0.0, "tilt_max": 0.0,
            })
            for i in range(len(self.vertices))
        ]

    # ── Click handler ─────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button == _PAN_BUTTON:
            self._start_pan(event)
            return
        if event.button != 1:
            return

        x, y = event.xdata, event.ydata

        if self._mode == "place" and self._place_idx is not None:
            self.vertices[self._place_idx] = (round(x, 3), round(y, 3))
            self._place_idx = None
            self._mode      = "add"
            self._rebuild_wall_props()
            self._redraw()
            self.verticesChanged.emit()

        elif self._mode == "add":
            if self._sym_active:
                self._handle_sym_click(x, y)
            else:
                self._save_undo()
                self.vertices.append((round(x, 3), round(y, 3)))
                self._rebuild_wall_props()
                self._redraw()
                self.verticesChanged.emit()

        elif self._mode == "select" and len(self.vertices) >= 2:
            idx = nearest_wall((x, y), self.vertices)
            if idx is not None:
                self._sel_wall = idx
                self._redraw()
                self.wallClicked.emit(idx)

        elif self._mode == "delete" and self.vertices:
            idx = _nearest_vertex((x, y), self.vertices, _VTOL)
            if idx is not None:
                self.delete_vertex(idx)

        elif self._mode == "add_audience":
            if self._sym_active:
                self._handle_aud_sym_click(x, y)
            else:
                self.add_audience_vertex(x, y)

        elif self._mode == "delete_audience" and self.audience_vertices:
            idx = _nearest_vertex((x, y), self.audience_vertices, _VTOL)
            if idx is not None:
                self.delete_audience_vertex(idx)

        elif self._mode == "place_source":
            if self._sym_active:
                self._handle_src_sym_click(x, y)
            else:
                self.add_source(x, y)

        elif self._mode == "delete_source" and self.sources:
            idx = _nearest_vertex((x, y), self.sources, _VTOL)
            if idx is not None:
                self.delete_source(idx)

    def _handle_sym_click(self, x: float, y: float):
        a, b     = _SYM_AXIS
        p        = (round(x, 3), round(y, 3))
        axis_len = math.hypot(b[0]-a[0], b[1]-a[1]) or 1.0
        dist     = abs(point_side(p, a, b)) / axis_len

        if dist <= self._axis_snap_tol():
            p         = project_point_on_line(p, a, b)
            side_sign = 0
        else:
            side_sign = 1 if point_side(p, a, b) > 0 else -1

        if side_sign != 0:
            if self._sym_side is None:
                self._sym_side = side_sign
            elif side_sign != self._sym_side:
                self.symStatusChanged.emit(
                    "⚠ That point is on the wrong side — "
                    "click on the highlighted half.")
                return

        self._save_undo()
        self._sym_chain.append(p)
        self._rebuild_sym_vertices()
        self._rebuild_wall_props()
        self._redraw()
        self.verticesChanged.emit()
        self.symStatusChanged.emit(self._status_text())

    def _handle_aud_sym_click(self, x: float, y: float):
        """Mirrored audience click — independent side lock from floor."""
        a, b     = _SYM_AXIS
        p        = (round(x, 3), round(y, 3))
        axis_len = math.hypot(b[0]-a[0], b[1]-a[1]) or 1.0
        dist     = abs(point_side(p, a, b)) / axis_len

        if dist <= self._axis_snap_tol():
            p         = project_point_on_line(p, a, b)
            side_sign = 0
        else:
            side_sign = 1 if point_side(p, a, b) > 0 else -1

        if side_sign != 0:
            if self._aud_sym_side is None:
                self._aud_sym_side = side_sign
            elif side_sign != self._aud_sym_side:
                self.symStatusChanged.emit(
                    "⚠ That point is on the wrong side — "
                    "click on the highlighted half.")
                return

        self._save_undo()
        self._aud_sym_chain.append(p)
        self._rebuild_aud_sym_vertices()
        self._redraw()
        self.audienceChanged.emit()
        self.symStatusChanged.emit(self._status_text())

    def _handle_src_sym_click(self, x: float, y: float):
        """Mirrored source click — independent side lock; mirrors source across axis."""
        a, b     = _SYM_AXIS
        p        = (round(x, 3), round(y, 3))
        axis_len = math.hypot(b[0]-a[0], b[1]-a[1]) or 1.0
        dist     = abs(point_side(p, a, b)) / axis_len

        if dist <= self._axis_snap_tol():
            p         = project_point_on_line(p, a, b)
            side_sign = 0
        else:
            side_sign = 1 if point_side(p, a, b) > 0 else -1

        if side_sign != 0:
            if self._src_sym_side is None:
                self._src_sym_side = side_sign
            elif side_sign != self._src_sym_side:
                self.symStatusChanged.emit(
                    "⚠ That point is on the wrong side — "
                    "click on the highlighted half.")
                return

        self.add_source(p[0], p[1])
        self.symStatusChanged.emit(self._status_text())

    # ── Scroll zoom ───────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        scale  = 1 / _ZOOM_STEP if event.button == "up" else _ZOOM_STEP
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        w, h   = xl[1]-xl[0], yl[1]-yl[0]
        nw, nh = w * scale, h * scale
        if nw < _ZOOM_MIN or nh < _ZOOM_MIN or nw > _ZOOM_MAX or nh > _ZOOM_MAX:
            return
        rx  = (xl[1] - event.xdata) / w
        ry  = (yl[1] - event.ydata) / h
        nxl = (event.xdata - nw*(1-rx), event.xdata + nw*rx)
        nyl = (event.ydata - nh*(1-ry), event.ydata + nh*ry)
        self.ax.set_xlim(nxl)
        self.ax.set_ylim(nyl)
        self._autofit  = False
        self._cur_xlim = nxl
        self._cur_ylim = nyl
        self.draw_idle()

    # ── Pan ───────────────────────────────────────────────────────────────────

    def _start_pan(self, event):
        self._panning      = True
        self._pan_start_px = (event.x, event.y)
        self._pan_xlim0    = self.ax.get_xlim()
        self._pan_ylim0    = self.ax.get_ylim()
        self.setCursor(Qt.ClosedHandCursor)

    def _on_motion(self, event):
        if not self._panning or event.x is None or event.y is None:
            return
        inv    = self.ax.transData.inverted()
        x0, y0 = inv.transform(self._pan_start_px)
        x1, y1 = inv.transform((event.x, event.y))
        dx, dy = x1 - x0, y1 - y0
        nxl = (self._pan_xlim0[0] - dx, self._pan_xlim0[1] - dx)
        nyl = (self._pan_ylim0[0] - dy, self._pan_ylim0[1] - dy)
        self.ax.set_xlim(nxl)
        self.ax.set_ylim(nyl)
        self._autofit  = False
        self._cur_xlim = nxl
        self._cur_ylim = nyl
        self.draw_idle()

    def _on_release(self, event):
        if event.button == _PAN_BUTTON and self._panning:
            self._panning = False
            self.unsetCursor()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _view_limits(self) -> tuple[tuple, tuple]:
        if not self.vertices or self._autofit or self._cur_xlim is None:
            all_pts = list(self.vertices) + list(self.audience_vertices) + list(self.sources)
            if all_pts:
                xs  = [p[0] for p in all_pts]
                ys  = [p[1] for p in all_pts]
                pad = 1.0
                return (min(xs)-pad, max(xs)+pad), (min(ys)-pad, max(ys)+pad)
            return self.ax.get_xlim(), self.ax.get_ylim()
        return self._cur_xlim, self._cur_ylim

    def _redraw(self):
        ax = self.ax
        ax.cla()
        self._style_ax()

        xlim, ylim = self._view_limits()

        if self._sym_active:
            self._draw_sym_axis(xlim, ylim)

        self._draw_audience()
        self._draw_sources()

        verts = self.vertices
        n     = len(verts)

        if n == 0:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            self.draw_idle()
            return

        xs, ys = [v[0] for v in verts], [v[1] for v in verts]

        if n >= 3:
            ax.add_patch(MplPolygon(
                list(zip(xs, ys)), closed=True,
                facecolor="#2a3f54", edgecolor="#aaaaaa",
                linewidth=1.5, alpha=0.6,
            ))

        for i in range(n):
            j        = (i + 1) % n
            x1, y1   = verts[i]
            x2, y2   = verts[j]
            selected = (self._sel_wall == i)
            ax.plot(
                [x1, x2], [y1, y2],
                color="#00bfff" if selected else "#aaaaaa",
                linewidth=3 if selected else 1.5,
                zorder=3,
            )
            mx, my = (x1+x2)/2, (y1+y2)/2
            length = math.hypot(x2-x1, y2-y1)
            tilt   = self.wall_props[i]["tilt_deg"] if self.wall_props else 0
            ax.text(mx, my, f" W{i+1}  {length:.2f}m  {tilt}°",
                    color="#888888", fontsize=7, zorder=4)

        in_delete = (self._mode == "delete")
        for i, (vx, vy) in enumerate(verts):
            placing = (self._mode == "place" and i == self._place_idx)
            color   = "#ffdd00" if placing else ("#ff4444" if in_delete else "#ffffff")
            size    = 60 if placing else (50 if in_delete else 30)
            ax.scatter([vx], [vy], color=color, s=size, zorder=5)
            label = (f" V{i+1} ← click to place" if placing
                     else f" V{i+1} ✕" if in_delete
                     else f" V{i+1}")
            ax.text(vx, vy, label,
                    color="#ffdd00" if placing else ("#ff4444" if in_delete else "#cccccc"),
                    fontsize=8, zorder=6)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self.draw_idle()

    def _draw_audience(self):
        ax    = self.ax
        verts = self.audience_vertices
        n     = len(verts)
        if n == 0:
            return

        in_del = (self._mode == "delete_audience")

        if n >= 3:
            xs, ys = [v[0] for v in verts], [v[1] for v in verts]
            ax.add_patch(MplPolygon(
                list(zip(xs, ys)), closed=True,
                facecolor="#5a8f3c", edgecolor="#8fd14f",
                linewidth=1.5, alpha=0.35, zorder=2,
            ))
            for i in range(n):
                j      = (i + 1) % n
                x1, y1 = verts[i]
                x2, y2 = verts[j]
                ax.plot([x1, x2], [y1, y2], color="#8fd14f",
                        linewidth=1.3, linestyle="-", zorder=3)
        elif n == 2:
            x1, y1 = verts[0]
            x2, y2 = verts[1]
            ax.plot([x1, x2], [y1, y2], color="#8fd14f",
                    linewidth=1.3, linestyle="-", zorder=3)

        for i, (vx, vy) in enumerate(verts):
            color = "#ff4444" if in_del else "#8fd14f"
            ax.scatter([vx], [vy], color=color, s=40, marker="s", zorder=5)
            label = f" A{i+1} ✕" if in_del else f" A{i+1}"
            ax.text(vx, vy, label, color=color, fontsize=7, zorder=6)

    def _draw_sources(self):
        if not self.sources:
            return
        ax       = self.ax
        in_del   = (self._mode == "delete_source")
        for i, (sx, sy) in enumerate(self.sources):
            color = "#ff4444" if in_del else "#ff9d00"
            ax.scatter([sx], [sy], color=color, s=90, marker="*",
                       edgecolor="#1e1e1e", linewidth=0.8, zorder=7)
            label = f"  S{i+1} ✕" if in_del else f"  S{i+1}"
            ax.text(sx, sy, label, color=color, fontsize=8,
                    fontweight="bold", zorder=8)

    def _draw_sym_axis(self, xlim: tuple, ylim: tuple):
        a, b   = _SYM_AXIS
        d      = np.array([b[0]-a[0], b[1]-a[1]], float)
        d     /= np.linalg.norm(d)
        diag   = math.hypot(xlim[1]-xlim[0], ylim[1]-ylim[0])
        A      = np.array(a, float)
        p1, p2 = A - d*diag*2, A + d*diag*2
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                     color="#ff66cc", linestyle="--", linewidth=1.4, zorder=2)

        # Determine which side is locked for the current mode
        if self._mode == "place_source":
            active_side = self._src_sym_side
        elif self._mode == "add_audience":
            active_side = self._aud_sym_side
        else:
            active_side = self._sym_side

        if active_side is not None:
            normal = np.array([-d[1], d[0]]) * active_side
            rect   = [(xlim[0], ylim[0]), (xlim[1], ylim[0]),
                      (xlim[1], ylim[1]), (xlim[0], ylim[1])]
            half   = clip_polygon_halfplane(rect, a, normal)
            if len(half) >= 3:
                self.ax.add_patch(MplPolygon(
                    half, closed=True,
                    facecolor="#ff66cc", edgecolor="none",
                    alpha=0.07, zorder=0,
                ))


# ═══════════════════════════════════════════════════════════════════════════════
# Source table
# ═══════════════════════════════════════════════════════════════════════════════

class SourceTable(QTableWidget):
    """
    Four-column table (#, X, Y, Z) for multiple sound sources.
    X and Y are synced from the canvas; X is locked during symmetric mode.
    Z is always freely editable.
    Emits sourcesEdited(list[(x,y,z)]) on any cell change.
    """

    sourcesEdited = Signal(list)
    _IDX, _X, _Y, _Z = 0, 1, 2, 3

    def __init__(self, parent=None):
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(["#", "X [m]", "Y [m]", "Z [m]"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self._updating  = False
        self._x_locked  = False
        self._z_values: list[float] = []
        self.itemChanged.connect(self._on_edit)

    def set_x_locked(self, locked: bool):
        self._x_locked = locked
        self._refresh_flags()

    def load_xy(self, sources_xy: list[tuple], default_z: float = 1.5):
        """Update X/Y from canvas; preserve or extend Z values."""
        self._updating = True
        n = len(sources_xy)
        while len(self._z_values) < n:
            self._z_values.append(default_z)
        self._z_values = self._z_values[:n]
        self.setRowCount(n)
        for i, (x, y) in enumerate(sources_xy):
            self._set_row(i, x, y, self._z_values[i])
        self._refresh_flags()
        self._updating = False

    def load_xyz(self, sources_xyz: list[tuple]):
        """Load full (x, y, z) data (e.g. from a JSON file)."""
        self._updating = True
        self._z_values = [float(z) for (_, _, z) in sources_xyz] if sources_xyz else []
        self.setRowCount(len(sources_xyz))
        for i, (x, y, z) in enumerate(sources_xyz):
            self._set_row(i, x, y, z)
        self._refresh_flags()
        self._updating = False

    def get_sources_xyz(self, canvas_sources: list[tuple]) -> list[tuple]:
        """Merge canvas XY (position truth) with table Z values."""
        result = []
        for i, (x, y) in enumerate(canvas_sources):
            z = self._z_values[i] if i < len(self._z_values) else 1.5
            result.append((float(x), float(y), float(z)))
        return result

    def _set_row(self, row: int, x: float, y: float, z: float):
        idx_item = QTableWidgetItem(str(row + 1))
        idx_item.setFlags(Qt.ItemIsEnabled)
        self.setItem(row, self._IDX, idx_item)
        self.setItem(row, self._X,   QTableWidgetItem(f"{x:.3f}"))
        self.setItem(row, self._Y,   QTableWidgetItem(f"{y:.3f}"))
        self.setItem(row, self._Z,   QTableWidgetItem(f"{z:.3f}"))

    def _refresh_flags(self):
        for r in range(self.rowCount()):
            for col, editable in [
                (self._X, not self._x_locked),
                (self._Y, True),
                (self._Z, True),
            ]:
                item = self.item(r, col)
                if item is None:
                    continue
                flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                if editable:
                    flags |= Qt.ItemIsEditable
                item.setFlags(flags)

    def _on_edit(self, item: QTableWidgetItem):
        if self._updating:
            return
        row, col = item.row(), item.column()
        try:
            val = float(item.text())
        except ValueError:
            return
        if col == self._Z:
            while len(self._z_values) <= row:
                self._z_values.append(1.5)
            self._z_values[row] = val
        sources_xyz = []
        for r in range(self.rowCount()):
            try:
                x = float(self.item(r, self._X).text())
                y = float(self.item(r, self._Y).text())
                z = float(self.item(r, self._Z).text())
                sources_xyz.append((x, y, z))
            except (ValueError, AttributeError):
                pass
        self.sourcesEdited.emit(sources_xyz)


# ═══════════════════════════════════════════════════════════════════════════════
# Vertex table
# ═══════════════════════════════════════════════════════════════════════════════

class VertexTable(QTableWidget):
    """Three-column table (#, X, Y) that emits verticesEdited on edit."""

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
            idx_item = QTableWidgetItem(str(i + 1))
            idx_item.setFlags(Qt.ItemIsEnabled)
            self.setItem(i, 0, idx_item)
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




# ═══════════════════════════════════════════════════════════════════════════════
# Wall inclination table
# ═══════════════════════════════════════════════════════════════════════════════

class WallInclinationTable(QTableWidget):
    """Two-column table (#, Inclination [deg]) synced with canvas wall_props.
    Emits tiltEdited(int, float) when the user changes a value."""

    tiltEdited = Signal(int, float)   # (wall_index, new_tilt_deg)

    def __init__(self, parent=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["#", "Inclination [°]"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self._updating = False
        self.itemChanged.connect(self._on_edit)

    def load(self, wall_props: list[dict]):
        self._updating = True
        self.setRowCount(len(wall_props))
        for i, w in enumerate(wall_props):
            idx_item = QTableWidgetItem(str(i + 1))
            idx_item.setFlags(Qt.ItemIsEnabled)
            self.setItem(i, 0, idx_item)
            tilt_item = QTableWidgetItem(f"{w['tilt_deg']:.2f}")
            self.setItem(i, 1, tilt_item)
        self._updating = False

    def _on_edit(self, item: QTableWidgetItem):
        if self._updating or item.column() != 1:
            return
        try:
            val = float(item.text())
        except ValueError:
            return
        val = max(-89.0, min(89.0, val))
        self.tiltEdited.emit(item.row(), val)

# ═══════════════════════════════════════════════════════════════════════════════
# Room Design Tab
# ═══════════════════════════════════════════════════════════════════════════════

class RoomDesignTab(QWidget):

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._suppress_cancel_sym_dlg = False
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        content = QHBoxLayout()
        content.setSpacing(10)

        # Left side: toolbar on top, canvas below
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(self._make_toolbar())
        left.addWidget(self._make_canvas_panel())
        content.addLayout(left, stretch=3)

        # Right side: control panel spans full height (toolbar + canvas)
        control = self._make_control_panel()
        control.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        content.addWidget(control, stretch=1)

        root.addLayout(content)

    def _make_toolbar(self) -> QFrame:
        """Top toolbar with Mode, Actions and View groups, matching the wireframe layout."""
        bar = QFrame()
        bar.setFrameShape(QFrame.StyledPanel)
        bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        outer = QHBoxLayout(bar)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(10)

        # Toolbar buttons: full width, taller to fit all button labels comfortably
        BTN_W, BTN_H = 215, 30

        def tb_btn(text, role, cb=None, checkable=False):
            b = QPushButton(text)
            b.setProperty("role", role)
            b.setFixedSize(BTN_W, BTN_H)
            if checkable:
                b.setCheckable(True)
            if cb:
                b.clicked.connect(cb)
            return b

        # ── Mode group  (3 cols × 3 rows) ────────────────────────────────
        # Col 0 (green)          Col 1 (red)                Col 2 (secondary)
        # Place Vertex           Delete Vertex              Add Vertex
        # Place Audience Vertex  Delete Audience Vertex     Select Wall
        # Place Source           Delete Source              Symmetric Room
        mode_frame = QGroupBox("Mode")
        mode_grid  = QGridLayout(mode_frame)
        mode_grid.setContentsMargins(4, 14, 4, 2)
        mode_grid.setSpacing(2)

        self.btn_add          = tb_btn("Place Vertex",          "success",   checkable=True)
        self.btn_del          = tb_btn("Delete Vertex",         "danger",    checkable=True)
        self.btn_new_vertex   = tb_btn("Add Vertex",            "secondary")
        self.btn_add_audience = tb_btn("Place Audience Vertex", "success",   checkable=True)
        self.btn_del_audience = tb_btn("Delete Audience Vertex","danger",    checkable=True)
        self.btn_sel          = tb_btn("Select Wall",           "secondary", checkable=True)
        self.btn_place_source = tb_btn("Place Source",          "success",   checkable=True)
        self.btn_del_source   = tb_btn("Delete Source",         "danger",    checkable=True)

        # Single symmetry button — text/color/callback change with state
        self.btn_sym = tb_btn("Symmetric Room", "secondary", self._on_sym_btn_clicked)

        # Row 0
        mode_grid.addWidget(self.btn_add,          0, 0)
        mode_grid.addWidget(self.btn_del,          0, 1)
        mode_grid.addWidget(self.btn_new_vertex,   0, 2)
        # Row 1
        mode_grid.addWidget(self.btn_add_audience, 1, 0)
        mode_grid.addWidget(self.btn_del_audience, 1, 1)
        mode_grid.addWidget(self.btn_sel,          1, 2)
        # Row 2
        mode_grid.addWidget(self.btn_place_source, 2, 0)
        mode_grid.addWidget(self.btn_del_source,   2, 1)
        mode_grid.addWidget(self.btn_sym,          2, 2)

        # Mode button group (exclusive check)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for b in (self.btn_add, self.btn_del,
                  self.btn_add_audience, self.btn_del_audience,
                  self.btn_sel, self.btn_place_source, self.btn_del_source):
            self._mode_group.addButton(b)

        self.btn_add.clicked.connect(          lambda: self._set_mode("add"))
        self.btn_sel.clicked.connect(          lambda: self._set_mode("select"))
        self.btn_del.clicked.connect(          lambda: self._set_mode("delete"))
        self.btn_add_audience.clicked.connect( lambda: self._set_mode("add_audience"))
        self.btn_del_audience.clicked.connect( lambda: self._set_mode("delete_audience"))
        self.btn_place_source.clicked.connect( lambda: self._set_mode("place_source"))
        self.btn_del_source.clicked.connect(   lambda: self._set_mode("delete_source"))
        self.btn_new_vertex.clicked.connect(self._insert_vertex)

        self.btn_add.setChecked(True)
        outer.addWidget(mode_frame)

        # ── Actions group  (2 cols × 3 rows) ─────────────────────────────
        # Col 0            Col 1
        # Undo             Redo
        # Save Room        Clear Room
        # Load Room        To Room Optimize
        act_frame = QGroupBox("Actions")
        act_grid  = QGridLayout(act_frame)
        act_grid.setContentsMargins(4, 14, 4, 2)
        act_grid.setSpacing(2)

        btn_undo     = tb_btn("Undo",             "secondary", None)
        btn_redo     = tb_btn("Redo",             "secondary", None)
        btn_save     = tb_btn("Save Room",        "success",   self._save_room_file)
        btn_clear    = tb_btn("Clear Room",       "danger",    None)
        btn_optimize = tb_btn("To Room Optimize", "success",   self._to_room_optimize)

        btn_load = tb_btn("Load Room", "secondary", self._load_room_file)

        self._tb_undo  = btn_undo
        self._tb_redo  = btn_redo
        self._tb_clear = btn_clear

        act_grid.addWidget(btn_undo,  0, 0)
        act_grid.addWidget(btn_redo,  0, 1)
        act_grid.addWidget(btn_save,  1, 0)
        act_grid.addWidget(btn_clear, 1, 1)
        act_grid.addWidget(btn_load,  2, 0, 1, 2)
        outer.addWidget(act_frame)

        # ── View group  (1 col × 2 rows) ─────────────────────────────────
        view_frame = QGroupBox("View")
        view_lay   = QVBoxLayout(view_frame)
        view_lay.setContentsMargins(4, 14, 4, 2)
        view_lay.setSpacing(2)

        btn_preview   = tb_btn("Preview 3D", "secondary", self._preview_3d)
        btn_zoom      = tb_btn("Reset Zoom", "secondary", None)
        self._tb_zoom = btn_zoom

        view_lay.addWidget(btn_preview)
        view_lay.addWidget(btn_zoom)
        view_lay.addStretch()
        outer.addWidget(view_frame)

        outer.addStretch()
        return bar

    def _make_canvas_panel(self) -> QGroupBox:
        box = QGroupBox(" Floor Plan")
        lay = QVBoxLayout(box)

        self.canvas = FloorCanvas()
        self.canvas.verticesChanged.connect(self._on_verts_changed)
        self.canvas.wallClicked.connect(self._on_wall_clicked)
        self.canvas.symModeChanged.connect(self._on_sym_mode_changed)
        self.canvas.symStatusChanged.connect(self._on_sym_status_changed)
        self.canvas.audienceChanged.connect(self._on_audience_changed)
        self.canvas.sourceChanged.connect(self._on_source_changed)
        lay.addWidget(self.canvas)

        # Connect toolbar buttons that depend on canvas
        self._tb_undo.clicked.connect(self.canvas.undo)
        self._tb_redo.clicked.connect(self.canvas.redo)
        self._tb_clear.clicked.connect(self.canvas.clear)
        self._tb_zoom.clicked.connect(self.canvas.reset_zoom)

        self.sym_status_lbl = QLabel("")
        self.sym_status_lbl.setStyleSheet("color: #ff66cc; font-size: 8pt;")
        self.sym_status_lbl.setWordWrap(True)
        lay.addWidget(self.sym_status_lbl)

        return box

    def _make_control_panel(self) -> QWidget:
        panel = QWidget()
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)


        info_box  = QGroupBox(" Room Info")
        info_grid = QGridLayout(info_box)
        info_grid.addWidget(QLabel("Floor area:"), 0, 0)
        self.area_lbl = QLabel("— m²")
        info_grid.addWidget(self.area_lbl, 0, 1)
        info_grid.addWidget(QLabel("Volume:"), 1, 0)
        self.vol_lbl = QLabel("— m³")
        info_grid.addWidget(self.vol_lbl, 1, 1)
        lay.addWidget(info_box)

        # ── Height ─────────────────────────────────────────────────────────
        h_box = QGroupBox(" Room Height [m]")
        h_lay = QHBoxLayout(h_box)
        self.height_edit = QLineEdit("3.0")
        self.height_edit.setValidator(QDoubleValidator(0.1, 100.0, 2, self))
        self.height_edit.textChanged.connect(self._update_room_info)
        h_lay.addWidget(self.height_edit)
        lay.addWidget(h_box)

        # ── Vertex table ──────────────────────────────────────────────────
        vt_box = QGroupBox(" Vertices")
        vt_lay = QVBoxLayout(vt_box)
        self.vtable = VertexTable()
        self.vtable.verticesEdited.connect(self.canvas.set_vertices)
        vt_lay.addWidget(self.vtable)
        lay.addWidget(vt_box)

        # ── Audience area ─────────────────────────────────────────────────
        aud_box = QGroupBox(" Audience Area")
        aud_lay = QVBoxLayout(aud_box)
        self.audience_table = VertexTable()
        self.audience_table.verticesEdited.connect(self.canvas.set_audience_vertices)
        aud_lay.addWidget(self.audience_table)
        lay.addWidget(aud_box)

        # ── Wall inclinations table ──────────────────────────────────────
        wall_box = QGroupBox(" Wall Inclinations")
        wall_lay = QVBoxLayout(wall_box)
        self.wall_table = WallInclinationTable()
        self.wall_table.tiltEdited.connect(self._on_wall_tilt_edited)
        wall_lay.addWidget(self.wall_table)
        lay.addWidget(wall_box)

        # ── Sources table ─────────────────────────────────────────────────
        src_box = QGroupBox(" Sources [m]")
        src_lay = QVBoxLayout(src_box)
        self.source_table = SourceTable()
        self.source_table.sourcesEdited.connect(self._on_sources_table_edited)
        src_lay.addWidget(self.source_table)
        lay.addWidget(src_box)

        lay.addStretch()

        # ── To Room Optimize ──────────────────────────────────────────────
        btn_opt = QPushButton("To Room Optimize")
        btn_opt.setProperty("role", "success")
        btn_opt.clicked.connect(self._to_room_optimize)
        lay.addWidget(btn_opt)
        return panel

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _btn(text: str, role: str, cb=None) -> QPushButton:
        b = QPushButton(text)
        b.setProperty("role", role)
        if cb is not None:
            b.clicked.connect(cb)
        return b

    def _set_mode(self, mode: str):
        self.canvas.set_mode(mode)
        btn_map = {
            "add":              self.btn_add,
            "select":           self.btn_sel,
            "delete":           self.btn_del,
            "add_audience":     self.btn_add_audience,
            "delete_audience":  self.btn_del_audience,
            "place_source":     self.btn_place_source,
            "delete_source":    self.btn_del_source,
        }
        btn_map[mode].setChecked(True)

    # ── Symmetry callbacks ────────────────────────────────────────────────────

    def _on_sym_btn_clicked(self):
        """Single symmetry button: starts or cancels symmetric mode depending on state."""
        if self.canvas._sym_active:
            self._cancel_symmetric_room()
        else:
            self._start_symmetric_room()

    def _update_sym_btn(self, active: bool):
        """Update the symmetry button's text and role to reflect current state."""
        if active:
            self.btn_sym.setText("Cancel Symmetry")
            self.btn_sym.setProperty("role", "danger")
        else:
            self.btn_sym.setText("Symmetric Room")
            self.btn_sym.setProperty("role", "secondary")
        # Force Qt to re-apply the stylesheet for the new role
        self.btn_sym.style().unpolish(self.btn_sym)
        self.btn_sym.style().polish(self.btn_sym)

    def _start_symmetric_room(self):
        if self.canvas.vertices:
            if QMessageBox.question(
                self, "Symmetric Room",
                "This will clear the current floor plan to start a symmetric "
                "room (axis: X = 0). Continue?",
                QMessageBox.Yes | QMessageBox.No,
            ) != QMessageBox.Yes:
                return
        self.canvas.start_symmetric_room()

    def _cancel_symmetric_room(self):
        if not self._suppress_cancel_sym_dlg:
            dlg = CancelSymmetryDialog(self)
            if dlg.exec() != QDialog.Accepted:
                return
            if dlg.dont_show_again():
                self._suppress_cancel_sym_dlg = True
        self.canvas.cancel_symmetric_room()
        self.state.is_symmetric = False

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_verts_changed(self):
        self.vtable.load(self.canvas.vertices)
        self.wall_table.load(self.canvas.wall_props)
        self._update_room_info()

    def _on_wall_clicked(self, idx: int):
        dlg = WallConfigDialog(self.canvas.wall_props[idx], self)
        if dlg.exec() == QDialog.Accepted:
            self.canvas.wall_props[idx] = dlg.get_wall()
            self.canvas.update_wall_tilt(idx, dlg.get_wall()["tilt_deg"])
            self.wall_table.load(self.canvas.wall_props)
            self._update_room_info()

    def _on_sym_mode_changed(self, active: bool):
        self._update_sym_btn(active)
        self.btn_new_vertex.setEnabled(not active)
        if active:
            self.btn_add.setChecked(True)
        # In symmetric mode X is managed by mirroring, not user-editable
        self.source_table.set_x_locked(active)
        self.state.is_symmetric = active

    def _on_sym_status_changed(self, text: str):
        self.sym_status_lbl.setText(text)

    def _on_wall_tilt_edited(self, idx: int, tilt: float):
        self.canvas.update_wall_tilt(idx, tilt)
        self.wall_table.load(self.canvas.wall_props)   # refresh (mirrors may update)
        self._update_room_info()

    def _on_audience_changed(self):
        self.audience_table.load(self.canvas.audience_vertices)

    def _on_source_changed(self):
        self.source_table.load_xy(self.canvas.sources)
        self._sync_sources_to_state()

    def _sync_sources_to_state(self):
        """Keep state.room_geometry sources in sync with canvas + table Z values."""
        sources_xyz = self._get_sources()
        data = self.state.room_geometry.setdefault("data", {})
        data["source_pos"] = [[x, y, z] for x, y, z in sources_xyz]

    def _on_sources_table_edited(self, sources_xyz: list):
        """User edited the table — push XY back to canvas (Z stays in table)."""
        new_xy = [(x, y) for (x, y, z) in sources_xyz]
        self.canvas._save_undo()
        self.canvas.sources = new_xy
        self.canvas._redraw()
        self._sync_sources_to_state()

    # ── Private getters ───────────────────────────────────────────────────────

    def _update_room_info(self):
        verts = self.canvas.vertices
        if len(verts) < 3:
            self.area_lbl.setText("— m²")
            self.vol_lbl.setText("— m³")
            return
        try:
            height = float(self.height_edit.text() or 0)
            tilts  = [w["tilt_deg"] for w in self.canvas.wall_props]
            floor, ceiling = compute_ceiling(verts, height, tilts)
            self.area_lbl.setText(f"{abs(signed_area(verts)):.3f} m²")
            self.vol_lbl.setText(f"{room_volume(floor, ceiling, height):.3f} m³")
        except (ValueError, ZeroDivisionError):
            self.area_lbl.setText("— m²")
            self.vol_lbl.setText("— m³")

    def _get_audience_area(self) -> list[tuple[float, float]]:
        return list(self.canvas.audience_vertices)

    def _get_sources(self) -> list[tuple[float, float, float]]:
        """Return list of (x, y, z) merging canvas XY with table Z values."""
        return self.source_table.get_sources_xyz(self.canvas.sources)

    def _insert_vertex(self):
        n = len(self.canvas.vertices)
        dlg = InsertVertexDialog(n, self)
        if dlg.exec() == QDialog.Accepted:
            self.canvas.insert_vertex(dlg.get_index())

    # ── Load / Save ───────────────────────────────────────────────────────────

    def _load_room_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Room", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            self.state.room_geometry = raw if "data" in raw else {"data": raw}
            self.refresh_from_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")

    def _save_room_file(self):
        verts = self.canvas.vertices
        if len(verts) < 3:
            QMessageBox.warning(self, "Save Room",
                                "At least 3 vertices are required.")
            return

        height     = float(self.height_edit.text())
        wall_props = self.canvas.wall_props
        verts, wall_props = _ccw_ordered(verts, wall_props)
        tilts      = [w["tilt_deg"] for w in wall_props]
        floor, ceiling = compute_ceiling(verts, height, tilts)
        volume = room_volume(floor, ceiling, height)
        is_sym = self.canvas._sym_active

        audience_area = self._get_audience_area()
        if len(audience_area) >= 3:
            audience_area, _ = _ccw_ordered(audience_area)

        sources = self._get_sources()

        geom = build_geometry_dict(
            verts, wall_props, height, original_verts=verts,
            audience_area=audience_area, sources=sources,
        )
        if "sources" in geom.get("data", {}):
            geom["data"]["source_pos"] = geom["data"].pop("sources")
        geom["is_symmetric"] = is_sym
        geom["volume"]       = volume

        self.state.room_geometry = geom
        self.state.is_symmetric  = is_sym

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Room", "", "JSON Files (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        with open(path, "w") as f:
            json.dump(geom, f, indent=4)

    def _to_room_optimize(self):
        verts = self.canvas.vertices
        if len(verts) < 3:
            QMessageBox.warning(self, "Room Optimize",
                                "At least 3 vertices are required.")
            return
        height     = float(self.height_edit.text())
        wall_props = self.canvas.wall_props
        verts, wall_props = _ccw_ordered(verts, wall_props)
        tilts      = [w["tilt_deg"] for w in wall_props]
        floor, ceiling = compute_ceiling(verts, height, tilts)

        audience_area = self._get_audience_area()
        if len(audience_area) >= 3:
            audience_area, _ = _ccw_ordered(audience_area)
        sources = self._get_sources()

        geom = build_geometry_dict(
            floor, wall_props, height, original_verts=verts,
            audience_area=audience_area, sources=sources,
        )
        if "sources" in geom.get("data", {}):
            geom["data"]["source_pos"] = geom["data"].pop("sources")
        geom["is_symmetric"] = self.canvas._sym_active
        geom["volume"]       = room_volume(floor, ceiling, height)
        self.state.room_geometry = geom

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
        """Reload canvas and tables from self.state.room_geometry."""
        geom = self.state.room_geometry.get("data", {})
        if not geom:
            return

        verts = [(v[0], v[1]) for v in geom.get("vertices", {}).values()]
        walls = geom.get("walls", {})
        wall_props = [{
            "id": f"W{i+1}", "tilt_deg": walls.get(f"W{i+1}", 0.0),
            "locked": False, "optimize_tilt": False,
            "tilt_min": 0.0, "tilt_max": 0.0,
        } for i in range(len(verts))]
        is_sym = self.state.room_geometry.get("is_symmetric", False)

        audience_area = [(v[0], v[1]) for v in geom.get("audience_area", {}).values()]

        # Support new 'sources' list and legacy 'source_pos' single entry
        raw_sources = geom.get("source_pos")
        if raw_sources:
            sources_xyz = [tuple(s) for s in raw_sources]
        elif geom.get("source_pos"):
            sp = geom["source_pos"]
            sources_xyz = [(sp[0], sp[1], sp[2])]
        else:
            sources_xyz = []
        sources_xy = [(x, y) for (x, y, *_) in sources_xyz]

        self.height_edit.setText(str(geom.get("Z", 3.0)))
        self.canvas.load_geometry(
            verts, wall_props, is_sym,
            audience_vertices=audience_area or None,
            sources=sources_xy or None,
        )
        self.vtable.load(self.canvas.vertices)
        self.wall_table.load(self.canvas.wall_props)
        self.audience_table.load(self.canvas.audience_vertices)
        self.source_table.load_xyz(sources_xyz)

        self._update_room_info()
        self.state.is_symmetric = is_sym
        self.btn_add.setChecked(True)
        self._on_sym_mode_changed(is_sym)

    # ── 3-D preview ───────────────────────────────────────────────────────────

    def _preview_3d(self):
        verts = self.canvas.vertices
        if len(verts) < 3:
            QMessageBox.warning(self, "Preview 3D",
                                "At least 3 vertices are required.")
            return
        height = float(self.height_edit.text())
        tilts  = [w["tilt_deg"] for w in self.canvas.wall_props]
        floor, ceiling = compute_ceiling(verts, height, tilts)
        n   = len(floor)
        pts = np.array([(x, y, 0.0) for x, y in floor] + list(ceiling))
        faces = (
            [[n] + list(reversed(range(n)))] +
            [[n] + list(range(n, 2*n))] +
            [[4, i, (i+1)%n, (i+1)%n + n, i+n] for i in range(n)]
        )
        mesh = pv.PolyData(pts, np.hstack(faces))
        pl   = pv.Plotter(off_screen=False)
        pl.set_background("#1e1e1e")
        pl.add_mesh(mesh, show_edges=True, color="silver", opacity=0.5)
        pl.add_axes()
        pl.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _dedupe(pts: list[tuple], tol: float = 1e-6) -> list[tuple]:
    """Remove consecutive duplicates (and first==last wrap-around)."""
    if not pts:
        return pts
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0]-out[-1][0]) > tol or abs(p[1]-out[-1][1]) > tol:
            out.append(p)
    if (len(out) > 1
            and abs(out[-1][0]-out[0][0]) <= tol
            and abs(out[-1][1]-out[0][1]) <= tol):
        out.pop()
    return out


def _nearest_vertex(
    click: tuple[float, float],
    verts: list[tuple[float, float]],
    tol:   float,
) -> int | None:
    """Return index of closest vertex within tol, or None."""
    if not verts:
        return None
    dists = [math.hypot(v[0]-click[0], v[1]-click[1]) for v in verts]
    idx   = min(range(len(dists)), key=lambda i: dists[i])
    return idx if dists[idx] <= tol else None


def _positive_x_indices(
    verts: list[tuple[float, float]],
    tol:   float = _AXIS_TOL,
) -> list[int]:
    """Indices of vertices with x >= -tol (positive half, including axis)."""
    return [i for i, (x, _y) in enumerate(verts) if x >= -tol]


def _ccw_ordered(
    verts:      list[tuple[float, float]],
    wall_props: list[dict] | None = None,
) -> tuple[list[tuple[float, float]], list[dict] | None]:
    """
    Return (verts, wall_props) re-wound counter-clockwise if needed.

    wall_props[i] is the wall on edge verts[i]→verts[i+1].  Reversing the
    vertex list (keeping verts[0] fixed) maps each original edge i to a new
    position; wall_props must be fully reversed (no anchor) to stay attached
    to the same physical wall.  Verified algebraically for n in [3, 8].
    """
    if len(verts) < 3 or signed_area(verts) >= 0:
        return list(verts), (list(wall_props) if wall_props is not None else None)
    new_verts = [verts[0]] + list(reversed(verts[1:]))
    new_walls = list(reversed(wall_props)) if wall_props is not None else None
    return new_verts, new_walls