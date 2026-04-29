from dolfinx.io import gmsh as gmshio
from room_modal_optimizer.meshing.mesher import Mesher
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import assemble_matrix
import ufl
import numpy as np
import matplotlib.pyplot as plt
from slepc4py import SLEPc

params = {
    # Plant lengths
    "Lx": 4,
    "Ly": 4,
    "Lz": 3,

    # Plant offsets
    "left_y0": 0.1,
    "left_y1": 0.2,
    "right_y0": -0.2,
    "right_y1": -0.3,
    "front_x0": 0.4,
    "front_x1": 0,
    "back_x0": 0.2,
    "back_x1": -0.3,

    # Wall inclination (degrees)
    "left_angle": 10,
    "right_angle": -10,
    "front_angle": 10,
    "back_angle": -10
}

# lc chosen from highest frequency:
# lambda_min = c / f_max = 343 / 200 = 1.715 m
# Use ~6 elems per wavelength:
# lc = 1.715 / 6 = 0.286 m
# Chosen: lc = 0.25 m
mesher = Mesher(params, lc=0.25)
mesh_path = mesher.create(visualize=False)

mesh_data = gmshio.read_from_msh(mesh_path, MPI.COMM_WORLD, 0, gdim=3)
domain = mesh_data.mesh
assert mesh_data.facet_tags is not None
facet_tags = mesh_data.facet_tags

rho0 = 1.225
c = 343.0

V = fem.functionspace(domain, ("Lagrange", 1))
p = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

k_form = fem.form(ufl.inner(ufl.grad(p), ufl.grad(v)) * ufl.dx)
m_form = fem.form(ufl.inner(p, v) * ufl.dx)

K = assemble_matrix(k_form, [])
M = assemble_matrix(m_form, [])

K.assemble()
M.assemble()

# Eigensolver
solver = SLEPc.EPS().create()
solver.setDimensions(20)
solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)

st = SLEPc.ST().create()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.1)
st.setFromOptions()

solver.setST(st)
solver.setOperators(K, M)

solver.solve()

####
xr, xi = K.createVecs()
tol, maxIt = solver.getTolerances()
nconv = solver.getConverged()

eig_vector = []
eig_freq = []

if nconv > 0:
    for i in range(nconv):
        k = solver.getEigenpair(i, xr , xi)
        fn = np.sqrt(k.real) / (2 * np.pi) * c
        eig_freq.append(fn)
        
        print("%12f Hz" % fn)
        vect = xr.getArray()
        eig_vector.append(vect.copy())
        
for i in range(nconv):
    with XDMFFile(domain.comm, "Mode_" + str(np.round(eig_freq[i])) + "_Hz.xdmf", "w") as xdmf:
        p = fem.Function(V)
        p.x.array[:] = eig_vector[i]
        p.x.scatter_forward()
        p.name = "p"
        
        xdmf.write_mesh(domain)
        xdmf.write_function(p)