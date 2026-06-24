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
        self.path = None
        self.patchSize = 1.0
        self.patchMetadata = []
        
        self.volume = None
        self.floor = None
        self.ceiling = None
        self.walls = []
        self.source = []

        # Patched surfaces
        self.ceilingRest = None
        self.ceilingPatches = []
        self.wallPatches = []
        
        self.floor_pts = None
        self.ceiling_pts = None
        
        self.intersection_validator = IntersectionValidator()
        self.intersection_error = False
    
    def create(self, params, lc=0.28, room_name='room', visualize=False, source_pos=None, patch=False):
        self.params = params['data']
        self.lc = lc
        self.room_name = room_name
        self.source_pos = source_pos
        self.patch = patch
        
        self.floor = None
        self.ceiling = None
        self.walls = []
        self.source = []
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

            if self.patch:
                self.buildPatchedGeometry()
            else:
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
            spheres = self.addSourceSphereVolume()

            out, out_map = gmsh.model.occ.cut(
                [(3, volume)],
                [(3, sphere) for sphere in spheres],
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
        spheres = []
        for source in self.source_pos:
            x, y, z = source
            spheres.append(gmsh.model.occ.addSphere(x, y, z, radius))
        
        return spheres
    
    def setTagsWithCenterOfMass(self):
        Lz = self.params["Z"]
        if self.patch:
            self.ceilingPatches = []
            self.wallPatches = []
            self.ceilingRest = None
            self.source = []

        for dim, tag in gmsh.model.getEntities(2):
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)

            is_source = False

            for source_x, source_y, source_z in self.source_pos:
                distance_to_source = ((cx - source_x)**2 + (cy - source_y)**2 + (cz - source_z)**2)**0.5

                if distance_to_source < 1e-6:
                    is_source = True
                    break

            if is_source:
                self.source.append(tag)
                continue

            if abs(cz - 0.0) < 1e-6:
                self.floor = tag
            elif abs(cz - Lz) < 1e-6:
                area = gmsh.model.occ.getMass(dim, tag)

                if self.patch:
                    targetArea = self.patchSize ** 2
                    tolerance = 0.20 * targetArea

                    if abs(area - targetArea) <= tolerance:
                        self.ceilingPatches.append(tag)
                    else:
                        self.ceilingRest = tag
                else:
                    self.ceiling = tag
            else:
                if self.patch:
                    self.wallPatches.append(tag)
                else:
                    self.walls.append(tag)


    def addPhysicalGroups(self):
        gmsh.model.addPhysicalGroup(3, [self.volume], 1)
        gmsh.model.setPhysicalName(3, 1, "Air")

        gmsh.model.addPhysicalGroup(2, [self.floor], 2)
        gmsh.model.setPhysicalName(2, 2, "Floor")

        if self.patch:
            if self.ceilingRest is not None:
                gmsh.model.addPhysicalGroup(2, [self.ceilingRest], 3)
                gmsh.model.setPhysicalName(2, 3, "CeilingRest")

            for counter, ceilingPatch in enumerate(self.ceilingPatches, start=1):
                physicalTag = 1000 + counter

                gmsh.model.addPhysicalGroup(2, [ceilingPatch], physicalTag)
                gmsh.model.setPhysicalName(
                    2,
                    physicalTag,
                    f"CeilingPatch_{counter}"
                )

            for counter, wallPatch in enumerate(self.wallPatches, start=1):
                physicalTag = 2000 + counter

                gmsh.model.addPhysicalGroup(2, [wallPatch], physicalTag)
                gmsh.model.setPhysicalName(
                    2,
                    physicalTag,
                    f"WallPatch_{counter}"
                )

        else:
            gmsh.model.addPhysicalGroup(2, [self.ceiling], 3)
            gmsh.model.setPhysicalName(2, 3, "Ceiling")

            gmsh.model.addPhysicalGroup(2, self.walls, 4)
            gmsh.model.setPhysicalName(2, 4, "Walls")

        if self.source:
            gmsh.model.addPhysicalGroup(2, self.source, 5)
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
    
    def buildPatchedGeometry(self):
        lc = self.lc
        factory = gmsh.model.occ

        ceilingPatchSurfaces = []
        wallPatchSurfaces = []
        ceilingRestSurfaces = []

        # Floor
        floor_pts = [factory.addPoint(*point, 0, lc) for point in self.floor_pts]
        floor_lines = [
            factory.addLine(floor_pts[i], floor_pts[(i + 1) % len(floor_pts)])
            for i in range(len(floor_pts))
        ]

        cl_floor = factory.addCurveLoop(floor_lines)
        floor = factory.addPlaneSurface([cl_floor])

        # Ceiling outer boundary
        ceiling_pts = [
            factory.addPoint(*point, self.params["Z"], lc)
            for point in self.ceiling_pts
        ]

        ceiling_lines = [
            factory.addLine(ceiling_pts[i], ceiling_pts[(i + 1) % len(ceiling_pts)])
            for i in range(len(ceiling_pts))
        ]

        cl_ceiling = factory.addCurveLoop(ceiling_lines)

        center_x = sum(p[0] for p in self.ceiling_pts) / len(self.ceiling_pts)
        center_y = sum(p[1] for p in self.ceiling_pts) / len(self.ceiling_pts)
        z = self.params["Z"]

        # Ceiling patches flood-fill
        created = {(0, 0)}
        originals = [(0, 0)]

        while originals:
            neighbours = []

            for i, j in originals:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue

                        neighbour = (i + di, j + dj)

                        if neighbour not in created and neighbour not in neighbours:
                            neighbours.append(neighbour)

            originals = []

            for neighbour in neighbours:
                i, j = neighbour

                patchCenterX = center_x + i * self.patchSize
                patchCenterY = center_y + j * self.patchSize

                patchCorners = [
                    (
                        patchCenterX - self.patchSize / 2,
                        patchCenterY - self.patchSize / 2,
                    ),
                    (
                        patchCenterX + self.patchSize / 2,
                        patchCenterY - self.patchSize / 2,
                    ),
                    (
                        patchCenterX + self.patchSize / 2,
                        patchCenterY + self.patchSize / 2,
                    ),
                    (
                        patchCenterX - self.patchSize / 2,
                        patchCenterY + self.patchSize / 2,
                    ),
                ]

                canCreate = all(
                    self.pointInPolygon(corner, self.ceiling_pts)
                    for corner in patchCorners
                )

                if canCreate:
                    created.add(neighbour)
                    originals.append(neighbour)

        pointTags = {}
        lineTags = {}
        edgeUseCount = {}
        edgeLineTags = {}

        for i, j in created:
            patchCorners = [
                (i - 0.5, j - 0.5),
                (i + 0.5, j - 0.5),
                (i + 0.5, j + 0.5),
                (i - 0.5, j + 0.5),
            ]

            # If point doesnt exist in dict, adds it
            for corner in patchCorners:
                if corner not in pointTags:
                    gx, gy = corner

                    x = center_x + gx * self.patchSize
                    y = center_y + gy * self.patchSize

                    pointTags[corner] = factory.addPoint(x, y, z, lc)

            patchLines = []

            # If line exists in dict, obtains it with corresponding direction
            cornerPairs = [
                (patchCorners[0], patchCorners[1]),
                (patchCorners[1], patchCorners[2]),
                (patchCorners[2], patchCorners[3]),
                (patchCorners[3], patchCorners[0]),
            ]

            for pointA, pointB in cornerPairs:
                if (pointA, pointB) in lineTags:
                    line = lineTags[(pointA, pointB)]

                elif (pointB, pointA) in lineTags:
                    line = -lineTags[(pointB, pointA)]

                else:
                    line = factory.addLine(
                        pointTags[pointA],
                        pointTags[pointB]
                    )

                    lineTags[(pointA, pointB)] = line

                edgeKey = tuple(sorted([pointA, pointB]))
                edgeUseCount[edgeKey] = edgeUseCount.get(edgeKey, 0) + 1
                edgeLineTags[edgeKey] = {
                    "line": line,
                    "pointA": pointA,
                    "pointB": pointB,
                }

                patchLines.append(line)

            cl_patch = factory.addCurveLoop(patchLines)
            patch = factory.addPlaneSurface([cl_patch])

            ceilingPatchSurfaces.append(patch)

        # Decide if ceiling rest is needed
        ceilingArea = 0.0

        for index in range(len(self.ceiling_pts)):
            x1, y1 = self.ceiling_pts[index]
            x2, y2 = self.ceiling_pts[(index + 1) % len(self.ceiling_pts)]

            ceilingArea += x1 * y2 - x2 * y1

        ceilingArea = abs(ceilingArea) / 2.0
        coveredCeilingArea = len(ceilingPatchSurfaces) * self.patchSize ** 2

        if coveredCeilingArea < ceilingArea - 1e-6:
            boundaryEdges = [
                edgeLineTags[edgeKey]
                for edgeKey, count in edgeUseCount.items()
                if count == 1
            ]

            # Order edge lines in order to create curve loop
            orderedLines = []
            orderedEdges = []

            currentEdge = boundaryEdges.pop(0)
            orderedLines.append(currentEdge["line"])
            orderedEdges.append((currentEdge["pointA"], currentEdge["pointB"], currentEdge["line"]))

            currentPoint = currentEdge["pointB"]

            while boundaryEdges:
                foundNext = False

                for index, edge in enumerate(boundaryEdges):
                    if edge["pointA"] == currentPoint:
                        orderedLines.append(edge["line"])
                        currentPoint = edge["pointB"]
                        boundaryEdges.pop(index)
                        foundNext = True
                        break

                    elif edge["pointB"] == currentPoint:
                        orderedLines.append(-edge["line"])
                        currentPoint = edge["pointA"]
                        boundaryEdges.pop(index)
                        foundNext = True
                        break

                if not foundNext:
                    raise RuntimeError("Could not order ceiling patch boundary lines")

            ceilingFull = factory.addPlaneSurface([cl_ceiling])

            out, _ = factory.cut(
                [(2, ceilingFull)],
                [(2, patch) for patch in ceilingPatchSurfaces],
                removeObject=True,
                removeTool=False
            )

            ceilingRestSurfaces = [
                tag
                for dim, tag in out
                if dim == 2
            ]

        # Walls
        for wallIndex in range(len(self.floor_pts)):
            floorA2d = self.floor_pts[wallIndex]
            floorB2d = self.floor_pts[(wallIndex + 1) % len(self.floor_pts)]

            ceilingA2d = self.ceiling_pts[wallIndex]
            ceilingB2d = self.ceiling_pts[(wallIndex + 1) % len(self.ceiling_pts)]

            A = (floorA2d[0], floorA2d[1], 0.0)
            B = (floorB2d[0], floorB2d[1], 0.0)
            C = (ceilingB2d[0], ceilingB2d[1], z)
            D = (ceilingA2d[0], ceilingA2d[1], z)

            bottomLength = math.dist(A, B)
            topLength = math.dist(D, C)
            leftLength = math.dist(A, D)
            rightLength = math.dist(B, C)

            wallWidth = (bottomLength + topLength) / 2
            wallHeight = (leftLength + rightLength) / 2

            nU = max(1, math.ceil(wallWidth / self.patchSize))
            nV = max(1, math.ceil(wallHeight / self.patchSize))

            pointTags = {}
            lineTags = {}

            for iu in range(nU + 1):
                for iv in range(nV + 1):
                    u = iu / nU
                    v = iv / nV

                    bottomX = A[0] + u * (B[0] - A[0])
                    bottomY = A[1] + u * (B[1] - A[1])
                    bottomZ = A[2] + u * (B[2] - A[2])

                    topX = D[0] + u * (C[0] - D[0])
                    topY = D[1] + u * (C[1] - D[1])
                    topZ = D[2] + u * (C[2] - D[2])

                    x = bottomX + v * (topX - bottomX)
                    y = bottomY + v * (topY - bottomY)
                    zz = bottomZ + v * (topZ - bottomZ)

                    pointTags[(iu, iv)] = factory.addPoint(x, y, zz, lc)

            for iu in range(nU):
                for iv in range(nV):
                    patchCorners = [
                        (iu, iv),
                        (iu + 1, iv),
                        (iu + 1, iv + 1),
                        (iu, iv + 1),
                    ]

                    cornerPairs = [
                        (patchCorners[0], patchCorners[1]),
                        (patchCorners[1], patchCorners[2]),
                        (patchCorners[2], patchCorners[3]),
                        (patchCorners[3], patchCorners[0]),
                    ]

                    patchLines = []

                    for pointA, pointB in cornerPairs:
                        if (pointA, pointB) in lineTags:
                            line = lineTags[(pointA, pointB)]

                        elif (pointB, pointA) in lineTags:
                            line = -lineTags[(pointB, pointA)]

                        else:
                            line = factory.addLine(
                                pointTags[pointA],
                                pointTags[pointB]
                            )

                            lineTags[(pointA, pointB)] = line

                        patchLines.append(line)

                    cl_wall_patch = factory.addCurveLoop(patchLines)
                    wallPatch = factory.addSurfaceFilling(cl_wall_patch)

                    wallPatchSurfaces.append(wallPatch)

        # Volume
        ceilingSurfaces = ceilingPatchSurfaces + ceilingRestSurfaces

        boundarySurfaces = (
            [floor]
            + ceilingSurfaces
            + wallPatchSurfaces
        )

        factory.synchronize()

        factory.removeAllDuplicates()

        factory.synchronize()

        boundarySurfaces = [
            tag
            for dim, tag in gmsh.model.getEntities(2)
        ]

        sl = factory.addSurfaceLoop(boundarySurfaces)
        volume = factory.addVolume([sl])

        sl = factory.addSurfaceLoop(boundarySurfaces)
        volume = factory.addVolume([sl])

        spheres = self.addSourceSphereVolume()

        out, out_map = gmsh.model.occ.cut(
            [(3, volume)],
            [(3, sphere) for sphere in spheres],
            removeObject=True,
            removeTool=True
        )

        self.volume = out[0][1]


    def pointInPolygon(self, point, polygon):
        x, y = point
        inside = False
        n = len(polygon)
        tol = 1e-9

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            # If point is on edge, count it as outside
            cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

            if abs(cross) < tol:
                if (
                    min(x1, x2) - tol <= x <= max(x1, x2) + tol and
                    min(y1, y2) - tol <= y <= max(y1, y2) + tol
                ):
                    return False

            if ((y1 > y) != (y2 > y)):
                x_intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)

                if x < x_intersection:
                    inside = not inside

        return inside