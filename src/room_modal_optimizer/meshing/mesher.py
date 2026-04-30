import gmsh
import math
from pathlib import Path

class Mesher:
    def __init__(self):
        self.params = None
        self.lc = None
        self.room_name = None
        self.source_pos = None
        
        self.volume = None
        self.floor = None
        self.ceiling = None
        self.walls = []
        self.source = None
        
        self.floor_pts = None
        self.ceiling_pts = None
    
    def create(self, params, lc=0.25, room_name='room', visualize=False, source_pos=None):
        self.params = params
        self.lc = lc
        self.room_name = room_name
        self.source_pos = source_pos
        
        self.floor = None
        self.ceiling = None
        self.walls = []
        self.source = None
        
        gmsh.initialize()
        gmsh.model.add(self.room_name)
        self.setFloorPoints()
        self.setCeilingPoints()
        self.buildGeometry()
        
        gmsh.model.occ.synchronize()
        if self.source_pos is not None:
            self.setTagsWithCenterOfMass()
        
        self.addPhysicalGroups()
        mesh_path = self.generateMesh()
        if visualize:
            gmsh.fltk.run()
        gmsh.finalize()
        
        print("volume:", self.volume)
        print("floor:", self.floor)
        print("ceiling:", self.ceiling)
        print("walls:", self.walls)
        print("source:", self.source)
        
        return mesh_path
        
    def setFloorPoints(self):
        Lx = self.params["Lx"]
        Ly = self.params["Ly"]

        # puntos base con offsets
        p_FL = (0 + self.params["front_x0"], 0 + self.params["left_y0"])
        p_FR = (Lx + self.params["front_x1"], 0 + self.params["right_y0"])
        p_BR = (Lx + self.params["back_x1"], Ly + self.params["right_y1"])
        p_BL = (0 + self.params["back_x0"], Ly + self.params["left_y1"])
        
        self.floor_pts = p_FL, p_FR, p_BR, p_BL

    def setCeilingPoints(self):
        Lz = self.params["Lz"]

        a_left  = math.radians(self.params["left_angle"])
        a_right = math.radians(self.params["right_angle"])
        a_front = math.radians(self.params["front_angle"])
        a_back  = math.radians(self.params["back_angle"])

        dx_left  = Lz * math.tan(a_left)
        dx_right = Lz * math.tan(a_right)
        dy_front = Lz * math.tan(a_front)
        dy_back  = Lz * math.tan(a_back)

        p_FL, p_FR, p_BR, p_BL = self.floor_pts

        t_FL = (p_FL[0] - dx_left,  p_FL[1] - dy_front)
        t_FR = (p_FR[0] + dx_right, p_FR[1] - dy_front)
        t_BR = (p_BR[0] + dx_right, p_BR[1] + dy_back)
        t_BL = (p_BL[0] - dx_left,  p_BL[1] + dy_back)

        self.ceiling_pts = t_FL, t_FR, t_BR, t_BL

    def buildGeometry(self):
        p_FL, p_FR, p_BR, p_BL = self.floor_pts
        t_FL, t_FR, t_BR, t_BL = self.ceiling_pts
        lc = self.lc
        
        factory = gmsh.model.occ
        
        # Floor
        p1 = factory.addPoint(*p_FL, 0, lc)
        p2 = factory.addPoint(*p_FR, 0, lc)
        p3 = factory.addPoint(*p_BR, 0, lc)
        p4 = factory.addPoint(*p_BL, 0, lc)

        l1 = factory.addLine(p1, p2)
        l2 = factory.addLine(p2, p3)
        l3 = factory.addLine(p3, p4)
        l4 = factory.addLine(p4, p1)
        
        cl_floor = factory.addCurveLoop([l1, l2, l3, l4])
        floor = factory.addPlaneSurface([cl_floor])
        
        # Ceiling
        p5 = factory.addPoint(*t_FL, self.params["Lz"], lc)
        p6 = factory.addPoint(*t_FR, self.params["Lz"], lc)
        p7 = factory.addPoint(*t_BR, self.params["Lz"], lc)
        p8 = factory.addPoint(*t_BL, self.params["Lz"], lc)

        l5 = factory.addLine(p5, p6)
        l6 = factory.addLine(p6, p7)
        l7 = factory.addLine(p7, p8)
        l8 = factory.addLine(p8, p5)

        cl_ceiling = factory.addCurveLoop([l5, l6, l7, l8])
        ceiling = factory.addPlaneSurface([cl_ceiling])
        
        # Walls
        l9  = factory.addLine(p1, p5)
        l10 = factory.addLine(p2, p6)
        l11 = factory.addLine(p3, p7)
        l12 = factory.addLine(p4, p8)
        
        w1 = factory.addSurfaceFilling(factory.addCurveLoop([ l1,  l10, -l5, -l9 ]))
        w2 = factory.addSurfaceFilling(factory.addCurveLoop([ l2,  l11, -l6, -l10]))
        w3 = factory.addSurfaceFilling(factory.addCurveLoop([ l3,  l12, -l7, -l11]))
        w4 = factory.addSurfaceFilling(factory.addCurveLoop([ l4,   l9, -l8, -l12]))
        
        walls = [w1, w2, w3, w4]

        # Volume
        sl = factory.addSurfaceLoop([floor, ceiling] + walls)
        volume = factory.addVolume([sl])
        
        if self.source_pos is not None:
            sphere = self.addSourceSphereVolume()
            out, out_map = gmsh.model.occ.cut(
                [(3, volume)],
                [(3, sphere)],
                removeObject=True,
                removeTool=True
            )
            self.volume = out[0][1]
            
        else:
            self.volume = volume
            self.floor = floor
            self.ceiling = ceiling
            self.walls = walls
            
    
    def addSourceSphereVolume(self, radius=0.3):
        x, y, z = self.source_pos
        sphere = gmsh.model.occ.addSphere(x, y, z, radius)
        
        return sphere
    
    def setTagsWithCenterOfMass(self):
        source_x, source_y, source_z = self.source_pos
        Lz = self.params["Lz"]

        for dim, tag in gmsh.model.getEntities(2):
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)
            distance_to_source = ((cx - source_x)**2 + (cy - source_y)**2 + (cz - source_z)**2)**0.5
            if distance_to_source < 1e-6:
                self.source = tag
                continue
            if abs(cz - 0.0) < 1e-6:
                self.floor = tag
            elif abs(cz - Lz) < 1e-6:
                self.ceiling = tag
            else:
                self.walls.append(tag)

    def addPhysicalGroups(self):
        gmsh.model.addPhysicalGroup(3, [self.volume], 1)
        gmsh.model.setPhysicalName(3, 1, "Air")

        gmsh.model.addPhysicalGroup(2, [self.floor], 2)
        gmsh.model.setPhysicalName(2, 2, "Floor")

        gmsh.model.addPhysicalGroup(2, [self.ceiling], 3)
        gmsh.model.setPhysicalName(2, 3, "Ceiling")

        gmsh.model.addPhysicalGroup(2, self.walls, 4)
        gmsh.model.setPhysicalName(2, 4, "Walls")
        
        if self.source is not None:
            gmsh.model.addPhysicalGroup(2, [self.source], 5)
            gmsh.model.setPhysicalName(2, 5, "Source")
        
    def generateMesh(self, dim=3):
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.lc)

        output = Path(f"data/{self.room_name}/mesh/{self.room_name}_mesh.msh")
        output.parent.mkdir(parents=True, exist_ok=True)

        gmsh.model.mesh.generate(dim)
        gmsh.write(str(output))

        return output