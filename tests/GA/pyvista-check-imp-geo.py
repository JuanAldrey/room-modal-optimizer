import numpy as np
import pyvista as pv
import math


params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [2.0, -0.2],
            "V3": [4.0, 0.3],
            "V4": [4.7, 1.6],
            "V5": [4.0, 3.0],
            "V6": [2.4, 3.5],
            "V7": [0.7, 3.0],
            "V8": [-0.4, 1.4],
        },
        "walls": {
            "W1": 0.22775101461778569,
            "W2": 7.625839293322446,
            "W3": -7.256793396480036,
            "W4": 1.7207176304230138,
            "W5": -5.2716140210033355,
            "W6": -6.959174512235528,
            "W7": -1.4234077869082995,
            "W8": 7.45011252919295,
        },
        "Z": 3.3700768177397533,
    }
}


def polygon_area(pts):
    area = 0.0
    n = len(pts)

    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return area / 2.0


def ensure_counter_clockwise(pts, angles):
    if polygon_area(pts) < 0:
        return pts[::-1], angles[::-1]

    return pts, angles


def line_intersection(p1, d1, p2, d2):
    """
    Intersección entre:
    p1 + t*d1
    p2 + u*d2
    """
    A = np.array([
        [d1[0], -d2[0]],
        [d1[1], -d2[1]],
    ])

    b = np.array([
        p2[0] - p1[0],
        p2[1] - p1[1],
    ])

    det = np.linalg.det(A)

    if abs(det) < 1e-9:
        return None

    t, u = np.linalg.solve(A, b)
    return p1 + t * d1


def compute_ceiling_points(floor_pts, wall_angles, Z, normal_sign=1):
    """
    normal_sign = 1 usa normal izquierda para polígono CCW.
    Si ves que el techo se desplaza al lado opuesto a tu Mesher, poné normal_sign = -1.
    """
    n = len(floor_pts)

    shifted_lines = []

    for i in range(n):
        p1 = np.array(floor_pts[i], dtype=float)
        p2 = np.array(floor_pts[(i + 1) % n], dtype=float)

        edge = p2 - p1
        length = np.linalg.norm(edge)
        direction = edge / length

        # Normal izquierda para polígono counter-clockwise
        normal = np.array([-direction[1], direction[0]]) * normal_sign

        angle_rad = math.radians(wall_angles[i])
        offset = Z * math.tan(angle_rad)

        shifted_p = p1 + offset * normal

        shifted_lines.append((shifted_p, direction))

    ceiling_pts = []

    for i in range(n):
        # vértice de techo i = intersección entre pared anterior y pared actual
        p_prev, d_prev = shifted_lines[i - 1]
        p_curr, d_curr = shifted_lines[i]

        inter = line_intersection(p_prev, d_prev, p_curr, d_curr)

        if inter is None:
            raise ValueError(f"Adjacent shifted walls are parallel at vertex {i + 1}")

        ceiling_pts.append(inter)

    return ceiling_pts, shifted_lines


def make_line_polydata(points, edges):
    lines = []

    for a, b in edges:
        lines.extend([2, a, b])

    poly = pv.PolyData()
    poly.points = np.array(points)
    poly.lines = np.array(lines)

    return poly


def plot_room(params):
    data = params["data"]
    Z = data["Z"]

    floor_pts = [
        data["vertices"][key]
        for key in sorted(data["vertices"].keys(), key=lambda k: int(k[1:]))
    ]

    wall_angles = [
        data["walls"][key]
        for key in sorted(data["walls"].keys(), key=lambda k: int(k[1:]))
    ]

    floor_pts, wall_angles = ensure_counter_clockwise(floor_pts, wall_angles)

    ceiling_pts, shifted_lines = compute_ceiling_points(
        floor_pts,
        wall_angles,
        Z,
        normal_sign=1,
    )

    n = len(floor_pts)

    # Puntos 3D
    floor_3d = [(x, y, 0.0) for x, y in floor_pts]
    ceiling_3d = [(x, y, Z) for x, y in ceiling_pts]

    points = floor_3d + ceiling_3d

    edges = []

    # piso
    for i in range(n):
        edges.append((i, (i + 1) % n))

    # techo
    for i in range(n):
        edges.append((n + i, n + ((i + 1) % n)))

    # paredes verticales/inclinadas
    for i in range(n):
        edges.append((i, n + i))

    room_wire = make_line_polydata(points, edges)

    plotter = pv.Plotter()
    plotter.add_mesh(room_wire.tube(radius=0.025), color="black")

    # Piso
    floor_poly = pv.PolyData(np.array(floor_3d))
    floor_poly.faces = np.hstack([[n], np.arange(n)])
    plotter.add_mesh(floor_poly, color="lightblue", opacity=0.25)

    # Techo
    ceiling_poly = pv.PolyData(np.array(ceiling_3d))
    ceiling_poly.faces = np.hstack([[n], np.arange(n)])
    plotter.add_mesh(ceiling_poly, color="salmon", opacity=0.35)

    # Labels
    floor_labels = [f"V{i + 1}" for i in range(n)]
    ceiling_labels = [f"C{i + 1}" for i in range(n)]

    plotter.add_point_labels(
        np.array(floor_3d),
        floor_labels,
        font_size=14,
        point_size=8,
    )

    plotter.add_point_labels(
        np.array(ceiling_3d),
        ceiling_labels,
        font_size=14,
        point_size=8,
    )

    # Ejes y vista
    plotter.add_axes()
    plotter.show_grid()
    plotter.view_isometric()
    plotter.show()


if __name__ == "__main__":
    plot_room(params)