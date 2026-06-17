from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
from .microphone import Microphone
import ufl
import numpy as np

class DirectSimulator:
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
        self.order = None
        
        #FEM - Direct
        self.direct_problem = None
        self.p_a = None

        # Frequency data
        self.freqs = np.arange(20, 201, 2)
        self.pressure_response = None
        self.spl_response = None

        # Physical constants
        self.rho0 = 1.225
        self.c = 343.0

        # Runtime parameters
        self.k = None
        self.omega = None

        # Microphone and source
        self.microphone = None
        self.source_strength = None
        
        # Room name
        self.room_name = None

    def simulate(self, mesh_path, mic_positions, order=1, room_name='room', freqs=None, use_impedance=True, wall_z=25.0 + 0j, floor_z=25.0 + 0j, ceiling_z=25.0 + 0j):
        self.room_name = room_name
        self.order = order
        self.use_impedance = use_impedance
        if freqs is not None:
            self.freqs = np.asarray(freqs, dtype=float)
        self.wall_z_value = wall_z
        self.floor_z_value = floor_z
        self.ceiling_z_value = ceiling_z

        self.loadMesh(mesh_path)
        self.setup()
        self.microphone = Microphone(self.domain, mic_positions)
        self.setupVariationalFormulation()
        self.computeFrequencyResponse()
        self.pressureToSpl()

        splResponse = np.asarray(self.spl_response)

        if splResponse.shape[0] == len(self.freqs):
            splResponse = splResponse.T

        self.spl_response = splResponse

        return self.freqs, self.spl_response
        
    def loadMesh(self, mesh_path):
        print("Loading mesh...")
        mesh_data = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, 0, gdim=3)
        self.domain = mesh_data.mesh
        assert mesh_data.facet_tags is not None
        self.tags = {
            name: tag
            for name, (dim, tag) in mesh_data.physical_groups.items()
        }
        self.facet_tags = mesh_data.facet_tags
        self.ds = ufl.Measure(
            "ds",
            domain=self.domain,
            subdomain_data=self.facet_tags
        )
        
    def setup(self):
        print("Initial setup...")
        self.V = fem.functionspace(self.domain, ("Lagrange", self.order))
        self.p = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        
        self.p_a = fem.Function(self.V)

        self.k = fem.Constant(self.domain, default_scalar_type(0))
        self.omega = fem.Constant(self.domain, default_scalar_type(0))
        self.source_strength = fem.Constant(self.domain, default_scalar_type(0.01))
        
        self.wall_z = fem.Constant(self.domain, default_scalar_type(self.wall_z_value))
        self.floor_z = fem.Constant(self.domain, default_scalar_type(self.floor_z_value))
        self.ceiling_z = fem.Constant(self.domain, default_scalar_type(self.ceiling_z_value))
        
    def setupVariationalFormulation(self):
        print("Setting up variational formulation...")
        
        a = ufl.inner(ufl.grad(self.p), ufl.grad(self.v)) * ufl.dx - self.k**2 * ufl.inner(self.p, self.v) * ufl.dx
        if self.use_impedance:
            a += 1j * self.k / self.floor_z * self.p * ufl.conj(self.v) * self.ds(self.tags["Floor"])
            a += 1j * self.k / self.ceiling_z * self.p * ufl.conj(self.v) * self.ds(self.tags["Ceiling"])
            a += 1j * self.k / self.wall_z * self.p * ufl.conj(self.v) * self.ds(self.tags["Walls"])
        L = - 1j * self.omega * self.rho0 * self.source_strength * ufl.conj(self.v) * self.ds(self.tags["Source"])
        
        self.direct_problem = LinearProblem(
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
        print("Computing frequency response...")
        self.pressure_response = np.zeros((len(self.freqs), self.microphone.n_mics), dtype=complex)
        
        for nf in range(0, len(self.freqs)):
            freq = self.freqs[nf]

            self.k.value = 2 * np.pi * freq / self.c
            self.omega.value = 2 * np.pi * freq

            self.direct_problem.solve()
            self.p_a.x.scatter_forward()

            p_f = self.microphone.listen(self.p_a)
            p_f = self.domain.comm.gather(p_f, root=0)

            if self.domain.comm.rank == 0:
                assert p_f is not None
                self.pressure_response[nf] = np.hstack(p_f).ravel()
                
    def pressureToSpl(self):
        print("Pressure to SPL...")
        eps = 1e-12
        p = np.maximum(np.abs(self.pressure_response), eps)

        self.spl_response = 20 * np.log10(
            p / np.sqrt(2) / 2e-5
        )