import math
import numpy as np


def signed_area(verts: list[tuple[float, float]]) -> float:
    a = 0.0
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def ensure_ccw(verts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return list(reversed(verts)) if signed_area(verts) < 0 else list(verts)


def inward_normal(v1: tuple, v2: tuple) -> np.ndarray:
    e = np.array([v2[0] - v1[0], v2[1] - v1[1]], float)
    e /= np.linalg.norm(e)
    return np.array([-e[1], e[0]])


def offset_line(v1: tuple, v2: tuple, offset: float):
    p = np.array(v1, float)
    d = np.array([v2[0] - v1[0], v2[1] - v1[1]], float)
    d /= np.linalg.norm(d)
    return p + offset * inward_normal(v1, v2), d


def line_intersect(p1: np.ndarray, d1: np.ndarray,
                   p2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    det = np.linalg.det(A)
    if abs(det) < 1e-9:
        raise ValueError("Parallel walls — reduce inclination.")
    t, _ = np.linalg.solve(A, p2 - p1)
    return p1 + t * d1


def project_point_on_line(p: tuple, a: tuple, b: tuple) -> tuple[float, float]:
    """Proyección ortogonal del punto p sobre la recta que pasa por a y b."""
    A = np.array(a, float)
    d = np.array([b[0] - a[0], b[1] - a[1]], float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return (float(p[0]), float(p[1]))
    d /= norm
    proj = A + np.dot(np.array(p, float) - A, d) * d
    return (float(proj[0]), float(proj[1]))


def reflect_point(p: tuple, a: tuple, b: tuple) -> tuple[float, float]:
    """Refleja el punto p respecto de la recta que pasa por a y b."""
    proj = np.array(project_point_on_line(p, a, b), float)
    refl = 2 * proj - np.array(p, float)
    return (float(refl[0]), float(refl[1]))


def point_side(p: tuple, a: tuple, b: tuple) -> float:
    """
    Signo (producto cruz) del lado de la recta a→b en el que está p.
    > 0: a un lado; < 0: al otro; ≈ 0: sobre la recta.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def clip_polygon_halfplane(
    poly_pts: list[tuple],
    point_on_line: tuple,
    normal: np.ndarray
) -> list[tuple]:
    """
    Recorta un polígono convexo `poly_pts` contra el semiplano
    {q : dot(q - point_on_line, normal) >= 0}.
    Sutherland–Hodgman para un solo borde (la recta de simetría).
    Usado para sombrear visualmente el lado activo en "Symmetric Room".
    """
    if not poly_pts:
        return []
    out = []
    p0  = np.array(point_on_line, float)
    nrm = np.array(normal, float)
    n   = len(poly_pts)
    for i in range(n):
        cur = np.array(poly_pts[i], float)
        nxt = np.array(poly_pts[(i + 1) % n], float)
        cur_in = np.dot(cur - p0, nrm) >= 0
        nxt_in = np.dot(nxt - p0, nrm) >= 0
        if cur_in:
            out.append((float(cur[0]), float(cur[1])))
        if cur_in != nxt_in:
            d = nxt - cur
            denom = np.dot(d, nrm)
            if abs(denom) > 1e-12:
                t = -np.dot(cur - p0, nrm) / denom
                inter = cur + t * d
                out.append((float(inter[0]), float(inter[1])))
    return out


def _polygon_centroid(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    return sum(xs)/len(xs), sum(ys)/len(ys)


def _inward_normal_robust(v1, v2, centroid):
    """Normal que siempre apunta hacia el centroide del polígono."""
    e = np.array([v2[0]-v1[0], v2[1]-v1[1]], float)
    e /= np.linalg.norm(e)
    n1 = np.array([-e[1],  e[0]])
    n2 = np.array([ e[1], -e[0]])
    mid = np.array([(v1[0]+v2[0])/2, (v1[1]+v2[1])/2])
    c   = np.array(centroid)
    return n1 if np.dot(n1, c - mid) > 0 else n2


def compute_ceiling(
    floor_verts: list[tuple[float, float]],
    height: float,
    tilts_deg: list[float]
) -> tuple[list, list]:
    verts = list(floor_verts)
    n = len(verts)
    centroid = _polygon_centroid(verts)

    lines = []
    for i in range(n):
        v1, v2 = verts[i], verts[(i+1) % n]
        normal = _inward_normal_robust(v1, v2, centroid)
        offset = height * math.tan(math.radians(tilts_deg[i]))
        p = np.array(v1, float) + offset * normal
        d = np.array([v2[0]-v1[0], v2[1]-v1[1]], float)
        d /= np.linalg.norm(d)
        lines.append((p, d))

    ceiling = []
    for i in range(n):
        p, d = lines[(i-1) % n]
        q, e = lines[i]
        c = line_intersect(p, d, q, e)
        ceiling.append((float(c[0]), float(c[1]), float(height)))

    return verts, ceiling


def pt_to_seg_dist(pt: tuple, a: tuple, b: tuple) -> float:
    p = np.array(pt, float)
    a = np.array(a, float)
    b = np.array(b, float)
    ab = b - a
    l2 = np.dot(ab, ab)
    if l2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / l2, 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def nearest_wall(click: tuple, verts: list[tuple], tol: float = 0.4) -> int | None:
    n = len(verts)
    dists = [pt_to_seg_dist(click, verts[i], verts[(i + 1) % n]) for i in range(n)]
    idx = int(np.argmin(dists))
    return idx if dists[idx] <= tol else None


def seg_intersect(p1, p2, p3, p4):
    """
    Intersección entre segmento p1-p2 y p3-p4.
    Devuelve el punto si se cruzan en el interior, None si no.
    """
    d1 = np.array(p2, float) - np.array(p1, float)
    d2 = np.array(p4, float) - np.array(p3, float)
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    eps = 1e-9
    if eps < t < 1 - eps and eps < u < 1 - eps:
        pt = np.array(p1, float) + t * d1
        return (round(float(pt[0]), 4), round(float(pt[1]), 4))
    return None


def find_intersection_with_polygon(new_v: tuple, vertices: list[tuple]):
    """
    Verifica si la arista (vertices[-1] -> new_v) cruza alguna arista existente.
    Devuelve (punto_interseccion, indice_arista) o (None, None).
    Solo verifica aristas no adyacentes al último vértice.
    """
    n = len(vertices)
    if n < 2:
        return None, None
    a, b = vertices[-1], new_v
    for i in range(n - 2):
        pt = seg_intersect(a, b, vertices[i], vertices[i + 1])
        if pt is not None:
            return pt, i
    return None, None


def build_geometry_dict(
    floor_verts: list[tuple[float, float]],
    wall_props: list[dict],
    height: float,
    original_verts: list[tuple[float, float]] | None = None,
    audience_area: list[tuple[float, float]] | None = None,
    sources: list[tuple[float, float, float]] | None = None,
) -> dict:
    """
    Devuelve:
    {
      "data": {
        "vertices":      {"V1": [x, y], ...},
        "walls":         {"W1": ang, ...},
        "audience_area": {"V1": [x, y], ...},   # opcional
        "Z":             h,
        "sources":       [[x, y, z], ...]        # opcional, lista de fuentes
      }
    }
    """
    verts_to_save = original_verts if original_verts is not None else floor_verts
    data = {
        "vertices": {f"V{i+1}": [float(x), float(y)] for i,(x,y) in enumerate(verts_to_save)},
        "walls":    {f"W{i+1}": float(w["tilt_deg"]) for i,w in enumerate(wall_props)},
        "Z":        float(height)
    }
    if audience_area is not None:
        data["audience_area"] = {
            f"V{i+1}": [float(x), float(y)] for i, (x, y) in enumerate(audience_area)
        }
    if sources is not None:
        data["sources"] = [[float(sx), float(sy), float(sz)] for sx, sy, sz in sources]
    return {"data": data}
def room_volume(
    floor_verts: list[tuple[float, float]],
    ceiling_verts: list[tuple[float, float, float]],
    height: float
) -> float:
    """
    Volumen exacto del recinto (incluso con paredes inclinadas, donde el
    techo no es paralelo al piso) usando la fórmula del prismatoide:

        V = h/6 * (A_floor + 4*A_mid + A_ceiling)

    donde A_mid es el área de la sección obtenida promediando cada vértice
    del piso con su correspondiente vértice del techo (sección a media
    altura). Para techo plano (todas las inclinaciones en 0) esto se
    reduce exactamente a A_floor * height.
    """
    if not floor_verts or height <= 0:
        return 0.0
    ceiling_xy = [(c[0], c[1]) for c in ceiling_verts]
    a_floor = abs(signed_area(floor_verts))
    a_ceil  = abs(signed_area(ceiling_xy))
    mid = [
        ((floor_verts[i][0] + ceiling_xy[i][0]) / 2,
         (floor_verts[i][1] + ceiling_xy[i][1]) / 2)
        for i in range(len(floor_verts))
    ]
    a_mid = abs(signed_area(mid))
    return height / 6.0 * (a_floor + 4 * a_mid + a_ceil)