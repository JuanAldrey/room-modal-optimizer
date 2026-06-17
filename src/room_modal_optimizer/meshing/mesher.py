import gmsh
import math
from pathlib import Path
from room_modal_optimizer.meshing.intersection_validator import IntersectionValidator

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
        
        self.intersection_validator = IntersectionValidator()
        self.intersection_error = False
    
    def create(self, params, lc=0.28, room_name='room', visualize=False, source_pos=None):
        self.params = params['data']
        self.lc = lc
        self.room_name = room_name
        self.source_pos = source_pos
        
        self.floor = None
        self.ceiling = None
        self.walls = []
        self.source = None
        self.intersection_error = False
        
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 0)
            
            gmsh.model.add(self.room_name)

            self.setFloorPoints()
            self.setCeilingPoints()

            if self.intersection_error:
                return None

            self.buildGeometry()

            gmsh.model.occ.synchronize()

            if self.source_pos is not None:
                self.setTagsWithCenterOfMass()

            self.addPhysicalGroups()
            mesh_path = self.generateMesh()

            if visualize:
                gmsh.fltk.run()

            return mesh_path

        finally:
            gmsh.finalize()
        
    def setFloorPoints(self):
        vertices = self.params["vertices"]
        walls = self.params["walls"]

        self.floor_pts = [
            tuple(vertices[key])
            for key in sorted(vertices.keys(), key=lambda k: int(k[1:]))
        ]

        self.wall_angles = [
            walls[key]
            for key in sorted(walls.keys(), key=lambda k: int(k[1:]))
        ]

        self.ensureCounterClockwise()

    def setCeilingPoints(self):
        Lz = self.params["Z"]

        shifted_walls = []
        n_walls = len(self.floor_pts)
        
        for i in range(n_walls):
            floor_pt1 = self.floor_pts[i]
            floor_pt2 = self.floor_pts[(i + 1) % n_walls]
            
            vec_x = floor_pt2[0] - floor_pt1[0]
            vec_y = floor_pt2[1] - floor_pt1[1]
            
            length = math.sqrt(vec_x**2 + vec_y**2)
            
            nx = vec_y / length
            ny = -vec_x / length
            
            d = Lz * math.tan(math.radians(self.wall_angles[i]))
            
            ceiling_pt1 = (floor_pt1[0] + d * nx, floor_pt1[1] + d * ny)
            ceiling_pt2 = (floor_pt2[0] + d * nx, floor_pt2[1] + d * ny)
            
            shifted_walls.append((ceiling_pt1, ceiling_pt2))
            
        ceiling_pts = []

        for i in range(n_walls):
            previous_wall = shifted_walls[i - 1]
            current_wall = shifted_walls[i]

            ceiling_pt = self.lineIntersection(
                previous_wall[0],
                previous_wall[1],
                current_wall[0],
                current_wall[1]
            )

            ceiling_pts.append(ceiling_pt)
            
        ceiling_crossings = self.intersection_validator.find_polygon_crossings(ceiling_pts)
        
        if ceiling_crossings:
            print("\nInvalid ceiling polygon: crossings detected")

            for crossing in ceiling_crossings:
                print(
                    f"Edge {crossing['edge_a']} crosses edge {crossing['edge_b']}"
                )
                print("points_a:", crossing["points_a"])
                print("points_b:", crossing["points_b"])
                
            self.intersection_error = True

        self.ceiling_pts = ceiling_pts

    def buildGeometry(self):
        lc = self.lc
        factory = gmsh.model.occ
        
        # Floor
        floor_pts = [factory.addPoint(*point, 0, lc) for point in self.floor_pts]
        floor_lines = [
            factory.addLine(floor_pts[i], floor_pts[(i + 1) % len(floor_pts)])
            for i in range(len(floor_pts))
        ]
     
        cl_floor = factory.addCurveLoop(floor_lines)
        floor = factory.addPlaneSurface([cl_floor])
        
        # Ceiling
        ceiling_pts = [factory.addPoint(*point, self.params["Z"], lc) for point in self.ceiling_pts]
        ceiling_lines = [
            factory.addLine(ceiling_pts[i], ceiling_pts[(i + 1) % len(ceiling_pts)])
            for i in range(len(ceiling_pts))
        ]
        
        """
        Print to check intersections in tests/meshing/check-polygon.py 
        print("floor_pts:")
        for i, p in enumerate(self.floor_pts):
            print(f"F{i+1}: {p}")

        print("ceiling_pts:")
        for i, p in enumerate(self.ceiling_pts):
            print(f"C{i+1}: {p}")
        """
        
        cl_ceiling = factory.addCurveLoop(ceiling_lines)
        ceiling = factory.addPlaneSurface([cl_ceiling])
        
        # Walls
        vertical_lines = [
            factory.addLine(floor_pts[i], ceiling_pts[i])
            for i in range(len(floor_pts))
        ]
        
        walls = []
        
        for i in range(len(floor_pts)):
            wall_loop = factory.addCurveLoop([
                floor_lines[i],
                vertical_lines[(i + 1) % len(floor_pts)],
                -ceiling_lines[i],
                -vertical_lines[i]
            ])

            wall = factory.addSurfaceFilling(wall_loop)
            walls.append(wall)

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
            
    
    def addSourceSphereVolume(self, radius=0.1):
        x, y, z = self.source_pos
        sphere = gmsh.model.occ.addSphere(x, y, z, radius)
        
        return sphere
    
    def setTagsWithCenterOfMass(self):
        source_x, source_y, source_z = self.source_pos
        Lz = self.params["Z"]

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
    
    def lineIntersection(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4)
            - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / den

        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4)
            - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / den

        return (px, py)
        
    def ensureCounterClockwise(self):
        if self.polygonSignedArea(self.floor_pts) < 0:
            self.floor_pts = [self.floor_pts[0]] + list(reversed(self.floor_pts[1:]))
            self.wall_angles = self.wall_angles[::-1]
            
    def polygonSignedArea(self, points):
        area = 0.0
        n = len(points)

        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1

        return area / 2.0