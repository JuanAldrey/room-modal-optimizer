from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import assemble_matrix
from slepc4py import SLEPc
from pathlib import Path
from dolfinx import geometry
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
        
        #FEM - Modal
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

    def simulate(self, mesh_path, order=2,room_name='room', export=False):
        self.room_name = room_name
        self.loadMesh(mesh_path)
        self.setup(order)
        self.computeModalAnalysis()
        self.obtainModes()
        pairs = sorted(zip(self.eig_freq, self.eig_vector), key=lambda x: x[0])

        self.eig_freq = [freq for freq, vec in pairs]
        self.eig_vector = [vec for freq, vec in pairs]

        #if export:
        #    self.exportModes()

        return self.eig_freq, self.eig_vector, self.n_modes

        
    def loadMesh(self, mesh_path):
        print("Loading mesh...")
        mesh_data = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, 0, gdim=3)
        self.domain = mesh_data.mesh
        assert mesh_data.facet_tags is not None
        self.facet_tags = mesh_data.facet_tags
        
    def setup(self, order):
        self.V = fem.functionspace(self.domain, ("Lagrange", order))
        self.p = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        
        k_form = fem.form(ufl.inner(ufl.grad(self.p), ufl.grad(self.v)) * ufl.dx)
        m_form = fem.form(ufl.inner(self.p, self.v) * ufl.dx)

        self.K = assemble_matrix(k_form, [])
        self.M = assemble_matrix(m_form, [])

        self.K.assemble()
        self.M.assemble()
    
    def computeModalAnalysis(self, target_freq=70.0, n_modes=160, tol=1e-8):
        self.n_modes = n_modes
        sigma = (2 * np.pi * target_freq / self.c) ** 2

        self.modal_problem = SLEPc.EPS().create(self.domain.comm)

        self.modal_problem.setOperators(self.K, self.M)
        self.modal_problem.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        self.modal_problem.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

        self.modal_problem.setDimensions(n_modes, max(2 * n_modes, n_modes + 20))

        self.modal_problem.setTolerances(tol, 1000)

        self.modal_problem.setTarget(sigma)
        self.modal_problem.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

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

        if nconv > 0:
            for i in range(nconv):
                k = self.modal_problem.getEigenpair(i, xr, xi)
                lam = k.real

                if not np.isfinite(lam):
                    continue

                if lam <= 1e-10:
                    continue

                fn = np.sqrt(lam) / (2 * np.pi) * self.c

                self.eig_freq.append(fn)

                xrNorm = xr.copy()
                self.normalizeModeMass(xrNorm)

                vect = xrNorm.getArray()
                self.eig_vector.append(vect.copy())

    def normalizeModeMass(self, vec):
        temp = vec.duplicate()
        self.M.mult(vec, temp)

        norm2 = vec.dot(temp)

        if norm2.real <= 0:
            return vec

        vec.scale(1.0 / np.sqrt(norm2.real))
        return vec
    
    def modeFunctionFromVector(self, modeVec):
        modeFunction = fem.Function(self.V)
        modeFunction.x.array[:] = modeVec
        modeFunction.x.scatter_forward()
        return modeFunction
    
    def computeLocalPoints(self, points):
        points = np.asarray(
            points,
            dtype=self.domain.geometry.x.dtype
        ).reshape(-1, 3)

        bbTree = geometry.bb_tree(self.domain, self.domain.topology.dim)

        cellCandidates = geometry.compute_collisions_points(bbTree, points)

        collidingCells = geometry.compute_colliding_cells(
            self.domain,
            cellCandidates,
            points
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
            np.asarray(validPoints, dtype=self.domain.geometry.x.dtype)
        )


    def evaluateFunctionAtPoints(self, function, points, localCells=None):
        points = np.asarray(
            points,
            dtype=self.domain.geometry.x.dtype
        ).reshape(-1, 3)

        if localCells is None:
            localCells, localPoints = self.computeLocalPoints(points)
        else:
            localPoints = points

        values = function.eval(localPoints, localCells)

        return np.asarray(values).reshape(-1)


    def buildPointCache(self, sourcePositions, receiverPositions):
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
                localCells=localCells
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

    def modalTransferFromCache(self, cache, sourceIndex, freqs, zeta=0.005):
        if self.eig_freq is None or len(self.eig_freq) == 0:
            raise RuntimeError("No hay frecuencias modales calculadas.")

        eigFreq = np.asarray(self.eig_freq, dtype=float)
        freqs = np.asarray(freqs, dtype=float)

        phiSources = cache["phi_sources"]
        phiReceivers = cache["phi_receivers"]

        phiSource = phiSources[sourceIndex, :]

        omegaN = 2.0 * np.pi * eigFreq
        omega = 2.0 * np.pi * freqs

        denominator = (
            omegaN[:, None] ** 2
            - omega[None, :] ** 2
            + 1j * 2.0 * zeta * omegaN[:, None] * omega[None, :]
        )

        modalWeights = phiReceivers * phiSource[None, :]

        H = modalWeights @ (1.0 / denominator)

        return H
    
    def modalTransferMultiSourceFromCache(self, cache, freqs, zeta=0.005):
        nSources = cache["phi_sources"].shape[0]

        responses = []

        for sourceIndex in range(nSources):
            H = self.modalTransferFromCache(
                cache=cache,
                sourceIndex=sourceIndex,
                freqs=freqs,
                zeta=zeta,
            )

            responses.append(H)

        return np.asarray(responses)

    '''
    def exportModes(self):
        output_dir = Path(f"data/{self.room_name}/modes")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(self.eig_freq)):
            filename = output_dir / f"Mode_{i}_{int(self.eig_freq[i])}_Hz.xdmf"

            with XDMFFile(self.domain.comm, str(filename), "w") as xdmf:
                p = fem.Function(self.V)
                p.x.array[:] = self.eig_vector[i]
                p.x.scatter_forward()
                p.name = "p"

                xdmf.write_mesh(self.domain)
                xdmf.write_function(p)
    '''    