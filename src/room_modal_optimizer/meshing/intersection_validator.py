import numpy as np

class IntersectionValidator:
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
        return len(self.find_polygon_crossings(pts)) > 0