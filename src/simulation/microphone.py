import numpy.typing as npt
import numpy as np
from dolfinx import (
    default_scalar_type,
    geometry,
)

class MicrophonePressure:
    def __init__(self, domain, microphone_position):
        """Initialize microphone(s).

        Args:
            domain: The domain to insert microphones on
            microphone_position: Position of the microphone(s).
                Assumed to be ordered as ``(mic0_x, mic1_x, ..., mic0_y, mic1_y, ..., mic0_z, mic1_z, ...)``

        """
        self._domain = domain
        self._position = np.asarray(
            microphone_position, dtype=domain.geometry.x.dtype
        ).reshape(3, -1)
        self._local_cells, self._local_position = self.compute_local_microphones()
        self.n_mics = self._position.shape[1]

    def compute_local_microphones(
        self,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.floating]]:
        """
        Compute the local microphone positions for a distributed mesh

        Returns:
            Two lists (local_cells, local_points) containing the local cell indices and the local points
        """
        points = self._position.T
        bb_tree = geometry.bb_tree(self._domain, self._domain.topology.dim)

        cells = []
        points_on_proc = []

        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(
            self._domain, cell_candidates, points
        )

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        return np.asarray(cells, dtype=np.int32), np.asarray(
            points_on_proc, dtype=self._domain.geometry.x.dtype
        )

    def listen(
    self,
    field,
    recompute_collisions: bool = False
    ) -> npt.NDArray[np.complexfloating]:

        if recompute_collisions:
            self._local_cells, self._local_position = self.compute_local_microphones()

        if len(self._local_cells) > 0:
            return field.eval(self._local_position, self._local_cells)
        else:
            return np.zeros(0, dtype=default_scalar_type)