from mpi4py import MPI
from dolfinx import mesh
from dolfinx import fem
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx import plot
import ufl
import numpy
import pyvista

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

# Function family to interpolate nodes
V = fem.functionspace(domain, ("Lagrange", 1))

uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)

# Set nodes on boundary to uD
tdim = domain.topology.dim # dimension of the mesh (2)
fdim = tdim - 1 # dimension of the boundary (1)
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology) # facet -> 1D entity (boundaries in this case are 1D)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets) # Identifies Dofs from V located in boundary facets
bc = fem.dirichletbc(uD, boundary_dofs) # Set uD to indentified DoFs

# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define de source term
f = fem.Constant(domain, default_scalar_type(-6))   # -> basically the laplacian of the boundary conditions 
                                                    #(in this specific example, uE = uD, such that laplacian(uE) = laplacian(uD) = -6)
                                                    
# Weak formulation
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * ufl.conj(v) * ufl.dx

# Forming and solving the linear system
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Poisson",
)
uh = problem.solve()

# Compute the error
V2 = fem.functionspace(domain, ("Lagrange", 2)) # Uses antoher (appropiate) higher order space 
uex = fem.Function(V2, name="u_exact")
uex.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
error_max = numpy.max(numpy.abs(uD.x.array - uh.x.array))
if domain.comm.rank == 0:  # Only print the error on one process
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")
    
    
# Plot mesh using pyvista
print(pyvista.global_theme.jupyter_backend)

domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")
    
# Plot function using pyvista
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()