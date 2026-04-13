import gmsh

gmsh.initialize()
gmsh.model.add("box")

lc = 0.1

# base 2D
p1 = gmsh.model.geo.addPoint(0,0,0,lc)
p2 = gmsh.model.geo.addPoint(1,0,0,lc)
p3 = gmsh.model.geo.addPoint(1,1,0,lc)
p4 = gmsh.model.geo.addPoint(0,1,0,lc)

l1 = gmsh.model.geo.addLine(p1,p2)
l2 = gmsh.model.geo.addLine(p2,p3)
l3 = gmsh.model.geo.addLine(p3,p4)
l4 = gmsh.model.geo.addLine(p4,p1)

cl = gmsh.model.geo.addCurveLoop([l1,l2,l3,l4])
s = gmsh.model.geo.addPlaneSurface([cl])

# EXTRUDE
out = gmsh.model.geo.extrude([(2, s)], 0, 0, 1)

gmsh.model.geo.synchronize()

# -----------------------------
# INTERPRETAR "out"
# -----------------------------
#[
# (2, top), -> (dim, entity)
# (3, volume),
# (2, wall1),
# (2, wall2),
# (2, wall3),
# (2, wall4)
#]

top_surface = out[0][1]     # techo
volume = out[1][1]          # volumen

# paredes → todo lo demás que sea superficie (dim=2)
walls = [ent[1] for ent in out[2:] if ent[0] == 2]

# -----------------------------
# PHYSICAL GROUPS
# -----------------------------

gmsh.model.addPhysicalGroup(3, [volume], 1)
gmsh.model.setPhysicalName(3, 1, "Air")

gmsh.model.addPhysicalGroup(2, walls, 2)
gmsh.model.setPhysicalName(2, 2, "Walls")

gmsh.model.addPhysicalGroup(2, [top_surface], 3)
gmsh.model.setPhysicalName(2, 3, "Ceiling")

gmsh.model.addPhysicalGroup(2, [s], 4)
gmsh.model.setPhysicalName(2, 4, "Floor")

# mesh
gmsh.model.mesh.generate(3)

gmsh.fltk.run()
gmsh.finalize()