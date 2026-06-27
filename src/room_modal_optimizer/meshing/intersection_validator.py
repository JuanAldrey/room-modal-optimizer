import numpy as np

class IntersectionValidator:
    """
    Validates self-intersections in 2D polygonal room contours.

    This helper is mainly used to detect invalid floor or ceiling polygons before
    building the Gmsh geometry. It checks whether non-adjacent polygon edges cross
    each other, including colinear or touching segment cases.
    """
    def _orientation(self, a, b, c):
        return (
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )


    def _on_segment(self, a, b, c, eps=1e-9):
        return (
            min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
        )


    def _segments_intersect(self, a, b, c, d, eps=1e-9):
        o1 = self._orientation(a, b, c)
        o2 = self._orientation(a, b, d)
        o3 = self._orientation(c, d, a)
        o4 = self._orientation(c, d, b)

        # Caso general: cruce real
        if o1 * o2 < -eps and o3 * o4 < -eps:
            return True

        # Casos colineales / tocar segmento
        if abs(o1) <= eps and self._on_segment(a, b, c, eps):
            return True

        if abs(o2) <= eps and self._on_segment(a, b, d, eps):
            return True

        if abs(o3) <= eps and self._on_segment(c, d, a, eps):
            return True

        if abs(o4) <= eps and self._on_segment(c, d, b, eps):
            return True

        return False


    def find_polygon_crossings(self, pts):
        """
        Finds crossings between non-adjacent edges of a 2D polygon.

        Adjacent edges are ignored because they naturally share a vertex. The first
        and last edges are also treated as adjacent. If crossings are found, the
        method returns metadata describing the crossing edge pairs.

        Args:
            pts (list[tuple[float, float]]): Polygon vertices as (x, y) points.

        Returns:
            list[dict]: List of detected crossings. Each dictionary contains the
            crossed edge indices and the corresponding edge endpoint coordinates.
        """
        pts = [np.array(p, dtype=float) for p in pts]
        n = len(pts)

        crossings = []

        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]

            for j in range(i + 1, n):
                # Ignorar edges vecinos
                if abs(i - j) <= 1:
                    continue

                # Ignorar primer y último edge, también son vecinos
                if i == 0 and j == n - 1:
                    continue

                c = pts[j]
                d = pts[(j + 1) % n]

                if self._segments_intersect(a, b, c, d):
                    crossings.append({
                        "edge_a": i + 1,
                        "edge_b": j + 1,
                        "points_a": (tuple(a), tuple(b)),
                        "points_b": (tuple(c), tuple(d)),
                    })

        return crossings


    def has_polygon_crossings(self, pts):
        """
        Checks whether a 2D polygon has any self-intersections.

        Args:
            pts (list[tuple[float, float]]): Polygon vertices as (x, y) points.

        Returns:
            bool: True if at least one crossing is detected, False otherwise.
        """
        return len(self.find_polygon_crossings(pts)) > 0