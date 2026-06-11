from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import LinearProblem, assemble_matrix
from .microphone import Microphone
from pathlib import Path
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
        
        # Impedance values
        self.wall_z = None
        self.floor_z = None
        self.ceiling_z = None

        # Runtime parameters
        self.k = None
        self.omega = None

        # Microphone and source
        self.microphone = None
        self.source_strength = None
        
        # Room name
        self.room_name = None

    def simulate(self, mesh_path, mic_positions, room_name='room', export=False):
        self.room_name = room_name
        self.loadMesh(mesh_path)
        self.setup()
        self.microphone = Microphone(self.domain, mic_positions)
        self.setupVariationalFormulation()
        self.computeFrequencyResponse(export)
        self.pressureToSpl()
    
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
        self.V = fem.functionspace(self.domain, ("Lagrange", 2))
        self.p = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        
        self.p_a = fem.Function(self.V)

        self.k = fem.Constant(self.domain, default_scalar_type(0))
        self.omega = fem.Constant(self.domain, default_scalar_type(0))
        self.source_strength = fem.Constant(self.domain, default_scalar_type(0.01))
        
        self.wall_z = fem.Constant(self.domain, default_scalar_type(50.0 + 0j))
        self.floor_z = fem.Constant(self.domain, default_scalar_type(100.0 + 0j))
        self.ceiling_z = fem.Constant(self.domain, default_scalar_type(50.0 + 0j))
        
    def setupVariationalFormulation(self):
        print("Setting up variational formulation...")
        
        a = ufl.inner(ufl.grad(self.p), ufl.grad(self.v)) * ufl.dx - self.k**2 * ufl.inner(self.p, self.v) * ufl.dx
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
    
    def computeFrequencyResponse(self, export):
        print("Computing frequency response...")
        self.pressure_response = np.zeros((len(self.freqs), self.microphone.n_mics), dtype=complex)
        
        if export:
            output = Path(f"data/{self.room_name}/direct/{self.room_name}_direct.xdmf")
            output.parent.mkdir(parents=True, exist_ok=True)
            xdmf = XDMFFile(self.domain.comm, str(output), "w")
            xdmf.write_mesh(self.domain)
        else:
            xdmf = None
        
        for nf in range(0, len(self.freqs)):
            freq = self.freqs[nf]

            self.k.value = 2 * np.pi * freq / self.c
            self.omega.value = 2 * np.pi * freq

            self.direct_problem.solve()
            self.p_a.x.scatter_forward()
            
            if export and freq % 20 == 0:
                self.p_a.name = "pressure"
                xdmf.write_function(self.p_a, float(freq))

            p_f = self.microphone.listen(self.p_a)
            p_f = self.domain.comm.gather(p_f, root=0)

            if self.domain.comm.rank == 0:
                assert p_f is not None
                self.pressure_response[nf] = np.hstack(p_f).ravel()
                
        if xdmf is not None:
            xdmf.close()
                
    def pressureToSpl(self):
        print("Pressure to SPL...")
        eps = 1e-12
        p = np.maximum(np.abs(self.pressure_response), eps)

        self.spl_response = 20 * np.log10(
            p / np.sqrt(2) / 2e-5
        )