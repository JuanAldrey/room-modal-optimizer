import itertools
import numpy as np


class GeneSpaceValidator:
    def validateGeneSpaceSafety(self, baseParams, geneSpaceConfig, margin=0.05):
        data = baseParams["data"]

        baseVertices = data["vertices"]
        audienceArea = data["audience_area"]
        sourcePos = np.asarray(data["source_pos"], dtype=float)

        zBase = float(data["Z"])
        zConfig = geneSpaceConfig.get("Z", {})
        zMin = float(zConfig.get("low", zBase))

        if sourcePos[2] <= margin or sourcePos[2] >= zMin - margin:
            return False, (
                f"source_pos z={sourcePos[2]} puede quedar fuera "
                f"para Z_min={zMin}"
            )

        vertexKeys = sorted(
            baseVertices.keys(),
            key=lambda key: int(key[1:])
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
        sourcePoint = sourcePos[:2]

        nChecked = 0

        for roomOption in itertools.product(*vertexOptions):
            roomPolygon = np.asarray(roomOption, dtype=float)
            nChecked += 1

            if self.polygonArea(roomPolygon) <= 1e-9:
                return False, (
                    f"Polígono inválido o clockwise "
                    f"en combinación extrema {nChecked}"
                )

            if self.polygonSelfIntersects(roomPolygon):
                return False, (
                    f"Polígono autointersectado "
                    f"en combinación extrema {nChecked}"
                )

            if not self.pointInsidePolygonWithMargin(
                sourcePoint,
                roomPolygon,
                margin,
            ):
                return False, (
                    f"source_pos={sourcePoint.tolist()} puede quedar fuera "
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

        return True, f"OK: {nChecked} combinaciones extremas validadas"

    def sortedPolygon(self, vertices):
        keys = sorted(
            vertices.keys(),
            key=lambda key: int(key[1:])
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