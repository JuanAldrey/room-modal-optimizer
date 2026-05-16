from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import assemble_matrix
from slepc4py import SLEPc
from pathlib import Path
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

        # Physical constants
        self.rho0 = 1.225
        self.c = 343.0
        
        # Room name
        self.room_name = None

    def simulate(self, mesh_path, room_name='room', export=False):
        self.room_name = room_name
        self.loadMesh(mesh_path)
        self.setup()
        self.computeModalAnalysis()
        self.obtainModes()
        if export:
            self.exportModes()
        
        return self.eig_freq, self.eig_vector
        
        
    def loadMesh(self, mesh_path):
        print("Loading mesh...")
        mesh_data = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, 0, gdim=3)
        self.domain = mesh_data.mesh
        assert mesh_data.facet_tags is not None
        self.facet_tags = mesh_data.facet_tags
        
    def setup(self):
        self.V = fem.functionspace(self.domain, ("Lagrange", 2))
        self.p = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        
        k_form = fem.form(ufl.inner(ufl.grad(self.p), ufl.grad(self.v)) * ufl.dx)
        m_form = fem.form(ufl.inner(self.p, self.v) * ufl.dx)

        self.K = assemble_matrix(k_form, [])
        self.M = assemble_matrix(m_form, [])

        self.K.assemble()
        self.M.assemble()
    
    # TODO: Investigate how to improve solver configuration    
    def computeModalAnalysis(self):
        self.modal_problem = SLEPc.EPS().create()
        self.modal_problem.setDimensions(80)
        self.modal_problem.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        st = SLEPc.ST().create()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(0.839)
        st.setFromOptions()

        self.modal_problem.setST(st)
        self.modal_problem.setOperators(self.K, self.M)

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

                vect = xr.getArray()
                self.eig_vector.append(vect.copy())
        
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