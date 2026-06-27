from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem
from .microphone import Microphone
from .resonator_impedances import Z_HELMHOLTZ, Z_MEMBRANA, Z_PERFORADO

import ufl
import numpy as np

class DirectSimulator:
    """
    Runs direct frequency-domain FEM simulations for room acoustic analysis.

    The class solves the Helmholtz equation on a Gmsh/DOLFINx room mesh and
    computes the complex pressure response at a set of microphone positions. It
    supports rigid or impedance boundary conditions, including patch-based
    impedance assignments for absorber optimization.

    The simulation pipeline loads the mesh, prepares the finite element function
    space, builds the variational formulation, solves the direct problem for each
    frequency, evaluates the pressure at the microphones, and converts the result
    to dB SPL.

    Physical groups defined in the mesh are used to identify the air volume,
    floor, ceiling, walls, source boundaries and optional absorber patches.
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

    def simulate(
            self, 
            mesh_path, 
            mic_positions, 
            order=1, 
            room_name='room', 
            freqs=None, 
            use_impedance=True, 
            wall_z=25.0 + 0j, 
            floor_z=25.0 + 0j, 
            ceiling_z=25.0 + 0j,
            patch=False,
            impedance_mappings=None
            ):
        """
        Runs the direct frequency-domain FEM simulation for a room mesh.

        This method loads the mesh, sets up the finite element space, creates the
        microphone receiver object, assembles the Helmholtz variational problem,
        computes the pressure response over the selected frequency range, and converts
        the result to SPL.

        The source strength is calibrated to produce 94 dB SPL at 1 m from a pulsating
        sphere with radius 0.10 m. Surface impedances can be applied either as global
        floor, wall and ceiling values or, in patch mode, through individual impedance
        mappings assigned to physical patch tags.

        Args:
            mesh_path (str | Path): Path to the Gmsh mesh file.
            mic_positions (list | np.ndarray): Microphone positions as [x, y, z]
                coordinates.
            order (int): Polynomial order of the finite element function space.
            room_name (str): Name used to identify the current room simulation.
            freqs (array-like | None): Frequencies to simulate in Hz. If None, the
                simulator uses the default frequency array.
            use_impedance (bool): If True, applies impedance boundary conditions.
                If False, boundaries are treated as rigid unless otherwise defined.
            wall_z (complex): Specific acoustic impedance assigned to wall surfaces.
            floor_z (complex): Specific acoustic impedance assigned to the floor.
            ceiling_z (complex): Specific acoustic impedance assigned to the ceiling.
            patch (bool): If True, enables patch-based impedance assignment.
            impedance_mappings (dict | None): Mapping between physical patch tags and
                impedance values used in patch mode.

        Returns:
            tuple[np.ndarray, np.ndarray]: Simulated frequencies and SPL response.
            The SPL response is returned with shape (n_mics, n_freqs).
        """
        self.room_name = room_name
        self.order = order
        self.use_impedance = use_impedance
        if freqs is not None:
            self.freqs = np.asarray(freqs, dtype=float)
        self.source_strength_values = self.calculateSourceVelocity94dBSPL(
            self.freqs,
            sourceRadius=0.10,
            refDistance=1.0,
        )
        self.wall_z_value = wall_z
        self.floor_z_value = floor_z
        self.ceiling_z_value = ceiling_z

        self.patch = patch
        self.impedance_mappings = impedance_mappings
        self.patch_z = {}

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
        """
        Loads a Gmsh mesh and extracts its physical boundary tags.

        The method imports the .msh file into DOLFINx, stores the computational
        domain, reads the physical groups defined in Gmsh, and creates the exterior
        facet integration measure used to apply boundary conditions.

        Args:
            mesh_path (str | Path): Path to the Gmsh .msh mesh file.

        Returns:
            None
        """
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
        """
        Initializes the finite element objects and simulation constants.

        The method creates the Lagrange function space, defines the trial and test
        functions, allocates the pressure solution function, and initializes the FEM
        constants used by the frequency-domain Helmholtz formulation.

        It also stores the global impedance values as DOLFINx constants so they can
        be used directly in the variational problem.

        Returns:
            None
        """
        print("Initial setup...")
        self.V = fem.functionspace(self.domain, ("Lagrange", self.order))
        self.p = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        
        self.p_a = fem.Function(self.V)

        self.k = fem.Constant(self.domain, default_scalar_type(0))
        self.omega = fem.Constant(self.domain, default_scalar_type(0))
        self.source_strength = fem.Constant(self.domain, default_scalar_type(0.0))

        self.wall_z = fem.Constant(self.domain, default_scalar_type(self.wall_z_value))
        self.floor_z = fem.Constant(self.domain, default_scalar_type(self.floor_z_value))
        self.ceiling_z = fem.Constant(self.domain, default_scalar_type(self.ceiling_z_value))
        
    def setupVariationalFormulation(self):
        """
        Builds the weak variational formulation for the direct Helmholtz simulation.

        The method defines the frequency-domain FEM problem using the pressure trial
        function and test function previously created in setup(). The base formulation
        corresponds to the Helmholtz equation in the room volume. If impedance
        boundary conditions are enabled, boundary terms are added for the floor,
        ceiling and walls.

        In patch mode, impedance terms are assigned individually to patch physical
        tags using self.impedance_mappings. Each patch impedance is stored as a
        DOLFINx Constant in self.patch_z so it can be updated later for each simulated
        frequency.

        The source is modeled as a Neumann boundary condition over the physical group
        named "Source". The resulting linear system is solved using PETSc with a
        direct LU factorization through MUMPS.

        Returns:
            None
        """
        print("Setting up variational formulation...")
        
        a = ufl.inner(ufl.grad(self.p), ufl.grad(self.v)) * ufl.dx - self.k**2 * ufl.inner(self.p, self.v) * ufl.dx
        if self.use_impedance:
            if self.patch:
                if "CeilingRest" in self.tags:
                    z = fem.Constant(self.domain, default_scalar_type(self.ceiling_z_value))
                    a += 1j * self.k / z * self.p * ufl.conj(self.v) * self.ds(self.tags["CeilingRest"])
                for tag, resonatorType  in self.impedance_mappings.items():
                    zInitial = self.get_impedance_value(resonatorType, freqIndex=0)
                    self.patch_z[tag] = fem.Constant(self.domain, default_scalar_type(zInitial))
                    a += 1j * self.k / self.patch_z[tag] * self.p * ufl.conj(self.v) * self.ds(tag)
            else:
                a += 1j * self.k / self.ceiling_z * self.p * ufl.conj(self.v) * self.ds(self.tags["Ceiling"])
                a += 1j * self.k / self.wall_z * self.p * ufl.conj(self.v) * self.ds(self.tags["Walls"])

            a += 1j * self.k / self.floor_z * self.p * ufl.conj(self.v) * self.ds(self.tags["Floor"])
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
        """
        Computes the complex pressure response at the microphone positions.

        The method iterates over all simulation frequencies, updates the wavenumber,
        angular frequency and source strength constants, and solves the direct FEM
        problem for each frequency. In patch mode, each patch impedance is also updated
        according to the current frequency before solving.

        After each solve, the pressure field is evaluated at the microphone positions.
        The microphone responses are gathered and stored as a frequency
        response matrix in self.pressure_response.

        Returns:
            None
        """
        print("Computing frequency response...")
        pressureRows = []
        
        for nf in range(0, len(self.freqs)):
            freq = self.freqs[nf]

            self.k.value = 2 * np.pi * freq / self.c
            self.omega.value = 2 * np.pi * freq

            self.source_strength.value = default_scalar_type(
                self.source_strength_values[nf]
            )

            if self.patch:
                for tag, resonatorType in self.impedance_mappings.items():
                    zValue = self.get_impedance_value(
                        resonatorType=resonatorType,
                        freqIndex=nf
                    )

                    self.patch_z[tag].value = default_scalar_type(zValue)

            self.direct_problem.solve()
            self.p_a.x.scatter_forward()

            p_f = self.microphone.listen(self.p_a)
            p_f = self.domain.comm.gather(p_f, root=0)

            if self.domain.comm.rank == 0:
                assert p_f is not None
                row = np.hstack(p_f).ravel()
                pressureRows.append(row)
                
        if self.domain.comm.rank == 0:
            self.pressure_response = np.vstack(pressureRows)

            print(
                f"Located microphones: {self.pressure_response.shape[1]} "
                f"of {self.microphone.n_mics}"
            )
            
    def pressureToSpl(self):
        """
        Converts the complex pressure response to sound pressure level in dB SPL.

        The method takes the magnitude of self.pressure_response, applies a small
        lower bound to avoid logarithm errors, converts peak pressure to RMS pressure,
        and computes SPL using 20 µPa as the reference pressure.

        The result is stored in self.spl_response.

        Returns:
            None
        """
        print("Pressure to SPL...")
        eps = 1e-12
        p = np.maximum(np.abs(self.pressure_response), eps)

        self.spl_response = 20 * np.log10(
            p / np.sqrt(2) / 2e-5
        )

    def calculateSourceVelocity94dBSPL(self, freqs, sourceRadius=0.10, refDistance=1.0):
        """
        Computes the source velocity needed to produce 94 dB SPL in free field.

        The source is modeled as a pulsating sphere with radius sourceRadius. For each
        frequency, the method estimates the normal surface velocity required to obtain
        94 dB SPL RMS at refDistance from the source in free-field conditions.

        The returned values are peak velocity amplitudes, obtained by converting the
        computed RMS velocities by a factor of sqrt(2).

        Args:
            freqs (array-like): Frequencies in Hz.
            sourceRadius (float): Radius of the spherical source in meters.
            refDistance (float): Reference distance from the source center in meters.

        Returns:
            np.ndarray: Peak normal velocity amplitudes for each frequency.
        """
        freqs = np.asarray(freqs, dtype=float)

        pRef = 2e-5
        pTargetRms = pRef * 10.0 ** (94.0 / 20.0)

        a = float(sourceRadius)
        r = float(refDistance)

        sourceVelocityRms = np.zeros_like(freqs, dtype=float)

        for i, freq in enumerate(freqs):
            k = 2.0 * np.pi * freq / self.c

            sourceVelocityRms[i] = (
                pTargetRms
                * r
                * np.sqrt(1.0 + (k * a) ** 2)
                / (self.rho0 * self.c * k * a ** 2)
            )

        return np.sqrt(2.0) * sourceVelocityRms
    
    def get_impedance_value(self, resonatorType, freqIndex):
        """
        Returns the impedance value for a resonator type at a given frequency index.

        The method maps integer resonator identifiers to their corresponding impedance
        arrays. A resonator type of 0 represents the default wall impedance, while
        types 1, 2 and 3 correspond to the predefined membrane, Helmholtz and
        perforated absorber impedance curves.

        Args:
            resonatorType (int): Resonator type identifier. Supported values are:
                0 for default impedance, 1 for membrane, 2 for Helmholtz and
                3 for perforated absorber.
            freqIndex (int): Index of the frequency value in the impedance arrays.

        Returns:
            complex: Complex impedance value for the selected resonator type and
            frequency index.

        Raises:
            ValueError: If the resonator type is not recognized.
        """
        resonatorType = int(resonatorType)

        if resonatorType == 0:
            return 25.0 + 0j

        if resonatorType == 1:
            return Z_MEMBRANA[freqIndex]

        if resonatorType == 2:
            return Z_HELMHOLTZ[freqIndex]

        if resonatorType == 3:
            return Z_PERFORADO[freqIndex]

        raise ValueError(f"Unknown resonator type: {resonatorType}")