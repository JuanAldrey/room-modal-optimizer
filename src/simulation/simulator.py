from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
from microphone import Microphone
import ufl
import numpy as np
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self):
        # Mesh / geometry
        self.domain = None
        self.facet_tags = None

        # FEM
        self.V = None
        self.problem = None
        self.p_a = None
        self.q = None

        # Frequency data
        self.freqs = np.arange(20, 200, 10)
        self.pressure_response = None
        self.spl_response = None

        # Physical constants
        self.rho0 = 1.225
        self.c = 343.0

        # Runtime parameters
        self.k = None
        self.omega = None

        # Microphone
        self.microphone = None

    def simulate(self, mesh_path, source, mic_pos):
        self.loadMesh(mesh_path)
        self.setup()
        self.microphone = Microphone(self.domain, mic_pos)
        self.setupBoundaryConditions()
        self.setupVariationalFormulation(source)
        self.computeFrequencyResponse()
        self.pressureToSpl()
        
        return self.spl_response
        
    def loadMesh(self, mesh_path):
        mesh_data = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, 0, gdim=3)
        self.domain = mesh_data.mesh
        assert mesh_data.facet_tags is not None
        self.facet_tags = mesh_data.facet_tags
        
    def setup(self):
        self.V = fem.functionspace(self.domain, ("Lagrange",1))
        self.q = fem.Function(self.V)
        self.p_a = fem.Function(self.V)

        self.k = fem.Constant(self.domain, default_scalar_type(0))
        self.omega = fem.Constant(self.domain, default_scalar_type(0))
    
    def setupBoundaryConditions(self):
        # MVP: only neuman conditions applied on variational formulation
        pass
    
    def setupMonopole(self, source_pos, radius=0.05, Q=1.0):
        xs, ys, zs = source_pos

        def source_field(x):
            r2 = (x[0] - xs)**2 + (x[1] - ys)**2 + (x[2] - zs)**2
            return np.where(r2 < radius**2, Q, 0.0)

        self.q.interpolate(source_field)
        
    def setupVariationalFormulation(self, source):
        p = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        self.setupMonopole(source)
        
        a = ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx - self.k**2 * ufl.inner(p, v) * ufl.dx
        L = 1j * self.omega * self.rho0 * ufl.conj(v) * self.q *ufl.dx
        
        self.problem = LinearProblem(
            a,
            L,
            u=self.p_a,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
            petsc_options_prefix="helmholtz",
        )
    
    def computeFrequencyResponse(self):
        self.pressure_response = np.zeros((len(self.freqs), self.microphone.n_mics), dtype=complex)
        for nf in range(0, len(self.freqs)):
            self.k.value = 2 * np.pi * self.freqs[nf] / self.c
            self.omega.value = 2 * np.pi * self.freqs[nf]

            self.problem.solve()
            self.p_a.x.scatter_forward()

            p_f = self.microphone.listen(self.p_a)
            p_f = self.domain.comm.gather(p_f, root=0)

            if self.domain.comm.rank == 0:
                assert p_f is not None
                self.pressure_response[nf] = np.hstack(p_f)
        
    def pressureToSpl(self):
        if self.domain.comm.rank == 0:
            self.spl_response = np.zeros_like(self.pressure_response.real)
            for m in range(self.microphone.n_mics):
                spl = 20 * np.log10(
                    np.abs(self.pressure_response[:, m]) / np.sqrt(2) / 2e-5
                )
                self.spl_response[:, m] = spl

                plt.plot(self.freqs, spl, label=f"Mic {m+1}")

            plt.grid(True)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("SPL [dB]")
            plt.xlim([self.freqs[0], self.freqs[-1]])
            plt.legend()
            plt.show()