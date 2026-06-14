from mpi4py import MPI
from dolfinx import fem, geometry
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import assemble_matrix
from slepc4py import SLEPc

import ufl
import numpy as np


class ModalSimulator:
    def __init__(self):
        # Mesh / geometry
        self.domain = None
        self.facet_tags = None

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
        self.room_name = room_name

        self.loadMesh(mesh_path)
        self.setup(order)

        self.computeModalAnalysis(
            target_freq=target_freq,
            n_modes=n_modes,
            tol=tol,
        )

        self.obtainModes()
        self.sortModes()

        if len(self.eig_freq) > 0:
            print("min eig freq:", min(self.eig_freq))
            print("max eig freq:", max(self.eig_freq))
            print("n eig freq:", len(self.eig_freq))

        return self.eig_freq, self.eig_vector, self.n_modes

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

    def computeModalAnalysis(self, target_freq=50.0, n_modes=160, tol=1e-8):
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

    # =========================================================
    # Cache modal para batch
    # =========================================================

    def buildPointCache(self, sourcePositions, receiverPositions):
        """
        Evalúa todos los modos en fuentes y receptores.

        Esta es la parte que permite hacer batch:
            una vez calculados los modos del recinto,
            se pueden evaluar muchas combinaciones fuente/mics.
        """
        if self.eig_vector is None or len(self.eig_vector) == 0:
            raise RuntimeError("No hay modos calculados. Ejecutar primero simulate().")

        sourcePositions = np.asarray(sourcePositions, dtype=float).reshape(-1, 3)
        receiverPositions = np.asarray(receiverPositions, dtype=float).reshape(-1, 3)

        nSources = sourcePositions.shape[0]
        nModes = len(self.eig_vector)

        allPoints = np.vstack([sourcePositions, receiverPositions])

        localCells, localPoints = self.computeLocalPoints(allPoints)

        phiAll = np.zeros((allPoints.shape[0], nModes), dtype=float)

        for modeIdx, modeVec in enumerate(self.eig_vector):
            modeFunction = self.modeFunctionFromVector(modeVec)

            values = self.evaluateFunctionAtPoints(
                modeFunction,
                localPoints,
                localCells=localCells,
            )

            phiAll[:, modeIdx] = np.real(values)

        phiSources = phiAll[:nSources, :]
        phiReceivers = phiAll[nSources:, :]

        return {
            "phi_sources": phiSources,
            "phi_receivers": phiReceivers,
            "source_positions": sourcePositions,
            "receiver_positions": receiverPositions,
        }

    # =========================================================
    # Transferencia modal rápida
    # =========================================================

    def modalTransferFromCache(
        self,
        cache,
        sourceIndex,
        freqs,
        zeta=0.0,
        sourceStrength=0.01,
    ):
        """
        Superposición modal rápida con fuente puntual.

        Modelo aproximado:
            fuente puntual idealizada en sourcePositions[sourceIndex]

        Ventaja:
            no requiere mallar la fuente
            no recalcula modos por cada posición de fuente
            sirve para batch/optimización

        Forma:
            p(r, f) = Σ phi_n(r) phi_n(xs) (-iωρ0Q)
                      / (k_n² - k² + i 2 zeta k_n k)
        """
        if self.eig_freq is None or len(self.eig_freq) == 0:
            raise RuntimeError("No hay frecuencias modales calculadas.")

        eigFreq = np.asarray(self.eig_freq, dtype=float)
        freqs = np.asarray(freqs, dtype=float)

        phiSources = cache["phi_sources"]
        phiReceivers = cache["phi_receivers"]

        phiSource = phiSources[sourceIndex, :]

        kN = 2.0 * np.pi * eigFreq / self.c
        k = 2.0 * np.pi * freqs / self.c
        omega = 2.0 * np.pi * freqs

        denominator = (
            kN[:, None] ** 2
            - k[None, :] ** 2
            + 1j * 2.0 * zeta * kN[:, None] * k[None, :]
        )

        modalWeights = phiReceivers * phiSource[None, :]

        H = modalWeights @ (1.0 / denominator)

        H = H * (-1j * omega[None, :] * self.rho0 * sourceStrength)

        return H

    def modalTransferMultiSourceFromCache(
        self,
        cache,
        freqs,
        zeta=0.0,
        sourceStrength=0.01,
    ):
        nSources = cache["phi_sources"].shape[0]

        responses = []

        for sourceIndex in range(nSources):
            H = self.modalTransferFromCache(
                cache=cache,
                sourceIndex=sourceIndex,
                freqs=freqs,
                zeta=zeta,
                sourceStrength=sourceStrength,
            )

            responses.append(H)

        return np.asarray(responses)