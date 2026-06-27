from mpi4py import MPI
from dolfinx import fem, geometry
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import assemble_matrix
from slepc4py import SLEPc

import ufl
import numpy as np


class ModalSimulator:
    """
    Unused class for transfer function analysis using modal superposition. Its use was discared after
    poor results in testing, but it is kept here for possible future use.
    """
    def __init__(self):
        # Mesh / geometry
        self.domain = None
        self.facet_tags = None
        self.tags = None
        self.ds = None

        # FEM
        self.V = None
        self.p = None
        self.v = None

        # FEM - Modal
        self.K = None
        self.M = None
        self.modal_problem = None
        self.eig_vector = []
        self.eig_freq = []
        self.n_modes = None
        self.source_weights = None

        # Physical constants
        self.rho0 = 1.225
        self.c = 343.0

        # Room name
        self.room_name = None

    # =========================================================
    # Main simulation helper
    # =========================================================

    def simulate(
        self,
        mesh_path,
        order=2,
        room_name="room",
        target_freq=50.0,
        n_modes=160,
        tol=1e-8,
    ):
        self.reset()

        self.room_name = room_name
        self.loadMesh(mesh_path)
        self.setup(order)

        print("pre MA...")
        self.computeModalAnalysis(
            target_freq=target_freq,
            n_modes=n_modes,
            tol=tol,
        )
        print("post MA...")

        self.obtainModes()
        self.sortModes()

        self.source_weights = self.computeSourceSurfaceWeights()

        return {
            "eig_freq": self.eig_freq,
            "eig_vector": self.eig_vector,
            "n_modes": self.n_modes,
            "source_weights": self.source_weights,
        }

    # =========================================================
    # Mesh / FEM setup
    # =========================================================

    def loadMesh(self, mesh_path):
        print("Loading mesh...")

        mesh_data = gmshio.read_from_msh(
            mesh_path,
            MPI.COMM_WORLD,
            0,
            gdim=3,
        )

        self.domain = mesh_data.mesh
        self.facet_tags = mesh_data.facet_tags

        if self.facet_tags is not None:
            self.tags = {
                name: tag
                for name, (dim, tag) in mesh_data.physical_groups.items()
            }

            self.ds = ufl.Measure(
                "ds",
                domain=self.domain,
                subdomain_data=self.facet_tags,
            )
        else:
            self.tags = {}
            self.ds = None

    def setup(self, order):
        self.V = fem.functionspace(self.domain, ("Lagrange", order))

        self.p = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        k_form = fem.form(
            ufl.inner(ufl.grad(self.p), ufl.grad(self.v)) * ufl.dx
        )

        m_form = fem.form(
            ufl.inner(self.p, self.v) * ufl.dx
        )

        self.K = assemble_matrix(k_form, [])
        self.M = assemble_matrix(m_form, [])

        self.K.assemble()
        self.M.assemble()

    # =========================================================
    # Modal analysis
    # =========================================================

    def computeModalAnalysis(self, target_freq=100.0, n_modes=160, tol=1e-8):
        self.n_modes = n_modes

        sigma = (2.0 * np.pi * target_freq / self.c) ** 2

        self.modal_problem = SLEPc.EPS().create(self.domain.comm)

        self.modal_problem.setOperators(self.K, self.M)
        self.modal_problem.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        self.modal_problem.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

        self.modal_problem.setDimensions(
            n_modes,
            max(2 * n_modes, n_modes + 20),
        )

        self.modal_problem.setTolerances(tol, 1000)

        self.modal_problem.setTarget(sigma)
        self.modal_problem.setWhichEigenpairs(
            SLEPc.EPS.Which.TARGET_MAGNITUDE
        )

        st = self.modal_problem.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(sigma)

        self.modal_problem.setFromOptions()
        self.modal_problem.solve()

    def obtainModes(self):
        xr, xi = self.K.createVecs()
        nconv = self.modal_problem.getConverged()

        self.eig_freq = []
        self.eig_vector = []

        for i in range(nconv):
            k = self.modal_problem.getEigenpair(i, xr, xi)
            lam = k.real

            if not np.isfinite(lam):
                continue

            if lam < -1e-8:
                continue

            if lam < 0.0:
                lam = 0.0

            fn = np.sqrt(lam) / (2.0 * np.pi) * self.c

            xrNorm = xr.copy()
            self.normalizeModeMass(xrNorm)

            modeVec = xrNorm.getArray().copy()

            self.eig_freq.append(fn)
            self.eig_vector.append(modeVec)

    def normalizeModeMass(self, vec):
        temp = vec.duplicate()
        self.M.mult(vec, temp)

        norm2 = vec.dot(temp)

        if norm2.real <= 0:
            return vec

        vec.scale(1.0 / np.sqrt(norm2.real))

        return vec

    def sortModes(self):
        pairs = sorted(
            zip(self.eig_freq, self.eig_vector),
            key=lambda pair: pair[0],
        )

        self.eig_freq = [freq for freq, vec in pairs]
        self.eig_vector = [vec for freq, vec in pairs]

    # =========================================================
    # Point evaluation
    # =========================================================

    def modeFunctionFromVector(self, modeVec):
        modeFunction = fem.Function(self.V)
        modeFunction.x.array[:] = modeVec
        modeFunction.x.scatter_forward()

        return modeFunction

    def computeLocalPoints(self, points):
        points = np.asarray(
            points,
            dtype=self.domain.geometry.x.dtype,
        ).reshape(-1, 3)

        bbTree = geometry.bb_tree(
            self.domain,
            self.domain.topology.dim,
        )

        cellCandidates = geometry.compute_collisions_points(
            bbTree,
            points,
        )

        collidingCells = geometry.compute_colliding_cells(
            self.domain,
            cellCandidates,
            points,
        )

        cells = []
        validPoints = []

        for i, point in enumerate(points):
            links = collidingCells.links(i)

            if len(links) == 0:
                raise ValueError(f"El punto {point} no cae dentro de la malla.")

            validPoints.append(point)
            cells.append(links[0])

        return (
            np.asarray(cells, dtype=np.int32),
            np.asarray(validPoints, dtype=self.domain.geometry.x.dtype),
        )

    def evaluateFunctionAtPoints(self, function, points, localCells=None):
        points = np.asarray(
            points,
            dtype=self.domain.geometry.x.dtype,
        ).reshape(-1, 3)

        if localCells is None:
            localCells, localPoints = self.computeLocalPoints(points)
        else:
            localPoints = points

        values = function.eval(localPoints, localCells)

        return np.asarray(values).reshape(-1)
    
    def computeSourceSurfaceWeights(self):
        if self.tags is None or "Source" not in self.tags:
            raise RuntimeError("La malla no tiene physical group 'Source'.")

        modeFunction = fem.Function(self.V)

        sourceForm = fem.form(
            modeFunction * self.ds(self.tags["Source"])
        )

        sourceWeights = []

        for modeVec in self.eig_vector:
            modeFunction.x.array[:] = modeVec
            modeFunction.x.scatter_forward()

            localValue = fem.assemble_scalar(sourceForm)

            totalValue = self.domain.comm.allreduce(
                localValue,
                op=MPI.SUM,
            )

            sourceWeights.append(totalValue)

        return np.asarray(sourceWeights, dtype=complex)
    
    def modalTransferFromFixedSurfaceSource(
        self,
        receiverPositions,
        freqs,
        sourceWeights,
        zeta=0.0,
        sourceStrength=0.01,
    ):
        if self.eig_freq is None or len(self.eig_freq) == 0:
            raise RuntimeError("No hay frecuencias modales calculadas.")

        receiverPositions = np.asarray(receiverPositions, dtype=float).reshape(-1, 3)
        freqs = np.asarray(freqs, dtype=float)
        sourceWeights = np.asarray(sourceWeights, dtype=complex)

        eigFreq = np.asarray(self.eig_freq, dtype=float)

        kN = 2.0 * np.pi * eigFreq / self.c
        k = 2.0 * np.pi * freqs / self.c
        omega = 2.0 * np.pi * freqs

        localCells, localPoints = self.computeLocalPoints(receiverPositions)

        nReceivers = receiverPositions.shape[0]
        nModes = len(self.eig_vector)

        phiReceivers = np.zeros((nReceivers, nModes), dtype=float)

        for modeIdx, modeVec in enumerate(self.eig_vector):
            modeFunction = self.modeFunctionFromVector(modeVec)

            values = self.evaluateFunctionAtPoints(
                modeFunction,
                localPoints,
                localCells=localCells,
            )

            phiReceivers[:, modeIdx] = np.real(values)

        denominator = (
            kN[:, None] ** 2
            - k[None, :] ** 2
            + 1j * 2.0 * zeta * kN[:, None] * k[None, :]
        )

        modalWeights = phiReceivers * sourceWeights[None, :]

        H = modalWeights @ (1.0 / denominator)

        H = H * (-1j * omega[None, :] * self.rho0 * sourceStrength)

        return H
    
    def reset(self):
        # Mesh / geometry
        self.domain = None
        self.facet_tags = None
        self.tags = None
        self.ds = None

        # FEM
        self.V = None
        self.p = None
        self.v = None

        # FEM - Modal
        self.K = None
        self.M = None
        self.modal_problem = None
        self.eig_vector = []
        self.eig_freq = []
        self.n_modes = None

        # Source
        self.source_weights = None

        # Room name
        self.room_name = None