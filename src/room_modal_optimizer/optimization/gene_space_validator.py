import itertools
import numpy as np


class GeneSpaceValidator:
    """
    Validates whether a geometry optimization gene space is safe.

    This helper class checks if the extreme combinations allowed by a GA gene
    space can generate invalid room geometries. It verifies that source positions
    and the audience area remain inside the room polygon with a minimum margin,
    and that the generated polygons do not become clockwise or self-intersecting.

    The validator supports both normal and symmetric optimization modes. In
    symmetric mode, only the master-side genes are tested and the mirrored
    geometry is reconstructed before validation.
    """
    def validateGeneSpaceSafety(
        self,
        baseParams,
        geneSpaceConfig,
        margin=0.05,
        keepSymmetry=False,
    ):
        """
        Validates the safety of a GA geometry search space.

        The method checks whether the extreme combinations defined by the gene space
        can produce invalid room geometries. It verifies source height limits, source
        positions, audience area placement, polygon orientation and polygon
        self-intersections.

        If keepSymmetry is True, the validation is performed using symmetric geometry
        reconstruction. Otherwise, all vertex extremes are tested directly.

        Args:
            baseParams (dict): Base room parameters containing geometry data under
                the "data" key.
            geneSpaceConfig (dict): Gene space configuration defining the allowed
                vertex, wall and height variations.
            margin (float): Minimum allowed distance between internal points or
                polygons and the room boundary.
            keepSymmetry (bool): If True, validates the search space assuming
                symmetric geometry reconstruction.

        Returns:
            tuple[bool, str]: Validation result and explanatory message.
        """
        data = baseParams["data"]

        baseVertices = data["vertices"]
        audienceArea = data["audience_area"]

        sourcePositions = np.asarray(data["source_pos"], dtype=float)

        if sourcePositions.ndim == 1:
            sourcePositions = sourcePositions.reshape(1, 3)

        if sourcePositions.ndim != 2 or sourcePositions.shape[1] != 3:
            return False, (
                "source_pos inválido: se esperaba [x, y, z] "
                "o [[x1, y1, z1], [x2, y2, z2], ...]"
            )

        zBase = float(data["Z"])
        zConfig = geneSpaceConfig.get("Z", {})
        zMin = float(zConfig.get("low", zBase))

        for i, sourcePos in enumerate(sourcePositions):
            if sourcePos[2] <= margin or sourcePos[2] >= zMin - margin:
                return False, (
                    f"source_pos[{i}] z={sourcePos[2]} puede quedar fuera "
                    f"para Z_min={zMin}"
                )

        if keepSymmetry:
            return self.validateSymmetricGeneSpaceSafety(
                baseVertices=baseVertices,
                audienceArea=audienceArea,
                sourcePositions=sourcePositions,
                geneSpaceConfig=geneSpaceConfig,
                margin=margin,
            )

        return self.validateNormalGeneSpaceSafety(
            baseVertices=baseVertices,
            audienceArea=audienceArea,
            sourcePositions=sourcePositions,
            geneSpaceConfig=geneSpaceConfig,
            margin=margin,
        )

    def validateNormalGeneSpaceSafety(
        self,
        baseVertices,
        audienceArea,
        sourcePositions,
        geneSpaceConfig,
        margin,
    ):
        """
        Validates all extreme vertex combinations for a non-symmetric gene space.

        Returns:
            tuple[bool, str]: Validation result and explanatory message.
        """
        vertexKeys = sorted(
            baseVertices.keys(),
            key=lambda key: int(key[1:]),
        )

        vertexOptions = []

        for key in vertexKeys:
            baseX, baseY = baseVertices[key]
            config = geneSpaceConfig.get("vertices", {}).get(key, {})

            dxMin, dxMax = config.get("dx", [0.0, 0.0])
            dyMin, dyMax = config.get("dy", [0.0, 0.0])

            options = [
                [baseX + dxMin, baseY + dyMin],
                [baseX + dxMin, baseY + dyMax],
                [baseX + dxMax, baseY + dyMin],
                [baseX + dxMax, baseY + dyMax],
            ]

            vertexOptions.append(options)

        audiencePolygon = self.sortedPolygon(audienceArea)
        sourcePoints = sourcePositions[:, :2]

        nChecked = 0

        for roomOption in itertools.product(*vertexOptions):
            roomPolygon = np.asarray(roomOption, dtype=float)
            nChecked += 1

            ok, message = self.validateRoomPolygon(
                roomPolygon=roomPolygon,
                audiencePolygon=audiencePolygon,
                sourcePoints=sourcePoints,
                margin=margin,
                nChecked=nChecked,
            )

            if not ok:
                return False, message

        return True, f"OK: {nChecked} combinaciones extremas validadas"

    def validateSymmetricGeneSpaceSafety(
        self,
        baseVertices,
        audienceArea,
        sourcePositions,
        geneSpaceConfig,
        margin,
    ):
        """
        Validates all extreme master-side combinations for a symmetric gene space.

        The mirrored vertices are reconstructed before each geometry validation.

        Returns:
            tuple[bool, str]: Validation result and explanatory message.
        """
        ok, message = self.validateSymmetricGeneKeys(
            baseVertices=baseVertices,
            geneSpaceConfig=geneSpaceConfig,
        )

        if not ok:
            return False, message

        masterKeys = sorted(
            geneSpaceConfig.get("vertices", {}).keys(),
            key=lambda key: int(key[1:]),
        )

        masterOptions = []

        for key in masterKeys:
            baseX, baseY = baseVertices[key]
            config = geneSpaceConfig["vertices"][key]

            dxMin, dxMax = config.get("dx", [0.0, 0.0])
            dyMin, dyMax = config.get("dy", [0.0, 0.0])

            options = [
                [baseX + dxMin, baseY + dyMin],
                [baseX + dxMin, baseY + dyMax],
                [baseX + dxMax, baseY + dyMin],
                [baseX + dxMax, baseY + dyMax],
            ]

            masterOptions.append(options)

        audiencePolygon = self.sortedPolygon(audienceArea)
        sourcePoints = sourcePositions[:, :2]

        nChecked = 0

        for masterOption in itertools.product(*masterOptions):
            vertices = self.copyVertices(baseVertices)

            for key, point in zip(masterKeys, masterOption):
                mirrorKey = self.findMirrorVertexKey(
                    masterKey=key,
                    baseVertices=baseVertices,
                )

                x, y = point

                vertices[key] = [
                    float(x),
                    float(y),
                ]

                vertices[mirrorKey] = [
                    float(-x),
                    float(y),
                ]

            roomPolygon = self.sortedPolygon(vertices)
            nChecked += 1

            ok, message = self.validateRoomPolygon(
                roomPolygon=roomPolygon,
                audiencePolygon=audiencePolygon,
                sourcePoints=sourcePoints,
                margin=margin,
                nChecked=nChecked,
            )

            if not ok:
                return False, message

        return True, f"OK simétrico: {nChecked} combinaciones extremas validadas"

    def validateSymmetricGeneKeys(self, baseVertices, geneSpaceConfig):
        selectedKeys = set(geneSpaceConfig.get("vertices", {}).keys())

        for key in selectedKeys:
            mirrorKey = self.findMirrorVertexKey(
                masterKey=key,
                baseVertices=baseVertices,
            )

            if mirrorKey in selectedKeys:
                return False, (
                    f"Gene space inválido para simetría: {key} y {mirrorKey} "
                    f"son espejos. Incluí solo uno de los dos."
                )

        return True, "OK"

    def validateRoomPolygon(
        self,
        roomPolygon,
        audiencePolygon,
        sourcePoints,
        margin,
        nChecked,
    ):
        area = self.polygonArea(roomPolygon)

        if area <= 1e-9:
            return False, (
                f"Polígono inválido o clockwise "
                f"en combinación extrema {nChecked}"
            )

        if self.polygonSelfIntersects(roomPolygon):
            return False, (
                f"Polígono autointersectado "
                f"en combinación extrema {nChecked}"
            )

        for i, sourcePoint in enumerate(sourcePoints):
            if not self.pointInsidePolygonWithMargin(
                sourcePoint,
                roomPolygon,
                margin,
            ):
                return False, (
                    f"source_pos[{i}]={sourcePoint.tolist()} puede quedar fuera "
                    f"o demasiado cerca del borde "
                    f"en combinación extrema {nChecked}"
                )

        if not self.polygonInsidePolygonWithMargin(
            audiencePolygon,
            roomPolygon,
            margin,
        ):
            return False, (
                f"audience_area puede quedar fuera "
                f"o demasiado cerca del borde "
                f"en combinación extrema {nChecked}"
            )

        return True, "OK"

    def findMirrorVertexKey(self, masterKey, baseVertices, axisTolerance=1e-9):
        masterPoint = np.asarray(baseVertices[masterKey], dtype=float)
        targetPoint = np.asarray(
            [-masterPoint[0], masterPoint[1]],
            dtype=float,
        )

        for vertexKey, point in baseVertices.items():
            if vertexKey == masterKey:
                continue

            point = np.asarray(point, dtype=float)

            if np.linalg.norm(point - targetPoint) <= axisTolerance:
                return vertexKey

        raise ValueError(
            f"No se encontró vértice espejo para {masterKey}. "
            f"Esperaba un punto cercano a {targetPoint.tolist()}"
        )

    def copyVertices(self, vertices):
        return {
            key: [float(value[0]), float(value[1])]
            for key, value in vertices.items()
        }

    def sortedPolygon(self, vertices):
        keys = sorted(
            vertices.keys(),
            key=lambda key: int(key[1:]),
        )

        return np.asarray(
            [vertices[key] for key in keys],
            dtype=float,
        )

    def polygonArea(self, points):
        x = points[:, 0]
        y = points[:, 1]

        return 0.5 * np.sum(
            x * np.roll(y, -1) - y * np.roll(x, -1)
        )

    def polygonInsidePolygonWithMargin(self, innerPolygon, outerPolygon, margin):
        for point in innerPolygon:
            if not self.pointInsidePolygonWithMargin(
                point,
                outerPolygon,
                margin,
            ):
                return False

        if self.polygonsIntersect(innerPolygon, outerPolygon):
            return False

        return True

    def pointInsidePolygonWithMargin(self, point, polygon, margin):
        if not self.pointInPolygon(point, polygon):
            return False

        distance = self.distancePointToPolygon(point, polygon)

        return distance >= margin

    def pointInPolygon(self, point, polygon):
        x, y = point

        inside = False
        n = len(polygon)
        j = n - 1

        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if (yi > y) != (yj > y):
                xIntersect = (
                    (xj - xi) * (y - yi)
                    / (yj - yi + 1e-12)
                    + xi
                )

                if x < xIntersect:
                    inside = not inside

            j = i

        return inside

    def distancePointToPolygon(self, point, polygon):
        distances = []

        for i in range(len(polygon)):
            a = polygon[i]
            b = polygon[(i + 1) % len(polygon)]

            distances.append(
                self.distancePointToSegment(point, a, b)
            )

        return float(np.min(distances))

    def distancePointToSegment(self, point, a, b):
        point = np.asarray(point, dtype=float)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        ab = b - a
        denom = np.dot(ab, ab)

        if denom < 1e-12:
            return float(np.linalg.norm(point - a))

        t = np.dot(point - a, ab) / denom
        t = np.clip(t, 0.0, 1.0)

        closest = a + t * ab

        return float(np.linalg.norm(point - closest))

    def polygonSelfIntersects(self, polygon):
        return self.polygonsIntersect(
            polygon,
            polygon,
            samePolygon=True,
        )

    def polygonsIntersect(self, polyA, polyB, samePolygon=False):
        for i in range(len(polyA)):
            a1 = polyA[i]
            a2 = polyA[(i + 1) % len(polyA)]

            for j in range(len(polyB)):
                if samePolygon:
                    if abs(i - j) <= 1:
                        continue

                    if i == 0 and j == len(polyA) - 1:
                        continue

                    if j == 0 and i == len(polyA) - 1:
                        continue

                b1 = polyB[j]
                b2 = polyB[(j + 1) % len(polyB)]

                if self.segmentsIntersect(a1, a2, b1, b2):
                    return True

        return False

    def segmentsIntersect(self, p1, p2, q1, q2):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)

        o1 = self.orientation(p1, p2, q1)
        o2 = self.orientation(p1, p2, q2)
        o3 = self.orientation(q1, q2, p1)
        o4 = self.orientation(q1, q2, p2)

        return (o1 * o2 < 0) and (o3 * o4 < 0)

    def orientation(self, a, b, c):
        ab = b - a
        ac = c - a

        return ab[0] * ac[1] - ab[1] * ac[0]