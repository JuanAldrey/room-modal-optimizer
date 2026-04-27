import gmsh
import math

class Mesher:
    def __init__(self, params, lc=0.5):
        self.params = params
        self.lc = lc
    
    def create(self):
        gmsh.initialize()
        gmsh.model.add("room")
        floor_pts = self.getFloorPoints()
        ceil_pts  = self.getCeilingPoints(floor_pts)
        volume, floor, ceiling, walls = self.buildGeometry(floor_pts, ceil_pts)
        gmsh.model.geo.synchronize()
        self.addPhysicalGroups(volume, floor, ceiling, walls)
        self.generateMesh()
        gmsh.fltk.run()
        gmsh.finalize()
        
    def getFloorPoints(self):
        Lx = self.params["Lx"]
        Ly = self.params["Ly"]

        # puntos base con offsets
        p_FL = (0 + self.params["front_x0"], 0 + self.params["left_y0"])
        p_FR = (Lx + self.params["front_x1"], 0 + self.params["right_y0"])
        p_BR = (Lx + self.params["back_x1"], Ly + self.params["right_y1"])
        p_BL = (0 + self.params["back_x0"], Ly + self.params["left_y1"])
        
        return (p_FL, p_FR, p_BR, p_BL)

    def getCeilingPoints(self, floor_points):
        Lz = self.params["Lz"]

        a_left  = math.radians(self.params["left_angle"])
        a_right = math.radians(self.params["right_angle"])
        a_front = math.radians(self.params["front_angle"])
        a_back  = math.radians(self.params["back_angle"])

        dx_left  = Lz * math.tan(a_left)
        dx_right = Lz * math.tan(a_right)
        dy_front = Lz * math.tan(a_front)
        dy_back  = Lz * math.tan(a_back)

        p_FL, p_FR, p_BR, p_BL = floor_points

        t_FL = (p_FL[0] - dx_left,  p_FL[1] - dy_front)
        t_FR = (p_FR[0] + dx_right, p_FR[1] - dy_front)
        t_BR = (p_BR[0] + dx_right, p_BR[1] + dy_back)
        t_BL = (p_BL[0] - dx_left,  p_BL[1] + dy_back)

        return (t_FL, t_FR, t_BR, t_BL)

    def buildGeometry(self, floor_Points, ceiling_points):
        p_FL, p_FR, p_BR, p_BL = floor_Points
        t_FL, t_FR, t_BR, t_BL = ceiling_points
        lc = self.lc
        
        # Floor
        p1 = gmsh.model.geo.addPoint(*p_FL, 0, lc)
        p2 = gmsh.model.geo.addPoint(*p_FR, 0, lc)
        p3 = gmsh.model.geo.addPoint(*p_BR, 0, lc)
        p4 = gmsh.model.geo.addPoint(*p_BL, 0, lc)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        
        cl_floor = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        floor = gmsh.model.geo.addPlaneSurface([cl_floor])
        
        # Ceiling
        p5 = gmsh.model.geo.addPoint(*t_FL, self.params["Lz"], lc)
        p6 = gmsh.model.geo.addPoint(*t_FR, self.params["Lz"], lc)
        p7 = gmsh.model.geo.addPoint(*t_BR, self.params["Lz"], lc)
        p8 = gmsh.model.geo.addPoint(*t_BL, self.params["Lz"], lc)

        l5 = gmsh.model.geo.addLine(p5, p6)
        l6 = gmsh.model.geo.addLine(p6, p7)
        l7 = gmsh.model.geo.addLine(p7, p8)
        l8 = gmsh.model.geo.addLine(p8, p5)

        cl_ceiling = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
        ceiling = gmsh.model.geo.addPlaneSurface([cl_ceiling])
        
        # Walls
        l9  = gmsh.model.geo.addLine(p1, p5)
        l10 = gmsh.model.geo.addLine(p2, p6)
        l11 = gmsh.model.geo.addLine(p3, p7)
        l12 = gmsh.model.geo.addLine(p4, p8)
        
        w1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([ l1,  l10, -l5, -l9 ])])
        w2 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([ l2,  l11, -l6, -l10])])
        w3 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([ l3,  l12, -l7, -l11])])
        w4 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop([ l4,   l9, -l8, -l12])])
        
        walls = [w1, w2, w3, w4]

        # Volume
        sl = gmsh.model.geo.addSurfaceLoop([floor, ceiling] + walls)
        volume = gmsh.model.geo.addVolume([sl])

        return volume, floor, ceiling, walls

    def addPhysicalGroups(self, volume, floor, ceiling, walls):
        gmsh.model.addPhysicalGroup(3, [volume], 1)
        gmsh.model.setPhysicalName(3, 1, "Air")

        gmsh.model.addPhysicalGroup(2, [floor], 2)
        gmsh.model.setPhysicalName(2, 2, "Floor")

        gmsh.model.addPhysicalGroup(2, [ceiling], 3)
        gmsh.model.setPhysicalName(2, 3, "Ceiling")

        gmsh.model.addPhysicalGroup(2, walls, 4)
        gmsh.model.setPhysicalName(2, 4, "Walls")
        
    def generateMesh(self, dim=3):
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.lc)

        gmsh.model.mesh.generate(dim)
        gmsh.write("data/mesh/mesh.msh")