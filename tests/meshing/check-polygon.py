import numpy as np
import matplotlib.pyplot as plt

def polygon_area(pts):
    area = 0.0
    n = len(pts)

    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return area / 2.0


def orient(a, b, c):
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def segments_intersect(a, b, c, d):
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    return o1 * o2 < 0 and o3 * o4 < 0


def check_polygon(name, pts):
    pts = [np.array(p, dtype=float) for p in pts]
    n = len(pts)

    print(f"\n--- {name} ---")
    print("Area:", polygon_area(pts))
    print("Orientation:", "CCW" if polygon_area(pts) > 0 else "CW")

    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        length = np.linalg.norm(b - a)
        print(f"Edge {i+1}: P{i+1} -> P{(i+1)%n + 1} | length = {length:.4f}")

    print("\nIntersections:")
    found = False

    for i in range(n):
        a1 = pts[i]
        a2 = pts[(i + 1) % n]

        for j in range(i + 1, n):
            # saltar lados vecinos
            if abs(i - j) <= 1:
                continue

            # saltar primer y último lado, también vecinos
            if i == 0 and j == n - 1:
                continue

            b1 = pts[j]
            b2 = pts[(j + 1) % n]

            if segments_intersect(a1, a2, b1, b2):
                print(
                    f"Intersection between edge {i+1} "
                    f"and edge {j+1}"
                )
                found = True

    if not found:
        print("No self-intersections found.")


# pegá acá tus puntos reales
floor_pts = [
    [0.0, 0.0],
    [2.0, -0.2],
    [4.0, 0.3],
    [4.7, 1.6],
    [4.0, 3.0],
    [2.4, 3.5],
    [0.7, 3.0],
    [-0.4, 1.4],
]

ceiling_pts = [
    [-0.46784943787765476, 0.03332198666633517],
    [3.2903855620042464, -0.34250151332185497],
    [3.1473979968930124, -0.3782484045996632],
    [4.524021798129037, 2.178338654838668],
    [4.327206119071877, 2.571970012952989],
    [2.5697960154615815, 3.1211606703312054],
    [0.4578772964611365, 2.500008105919311],
    [-0.6939397050769588, 0.824637921863899],
]


pts = np.array(ceiling_pts)

def segment_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-12:
        return None

    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den

    # chequeo de que el punto caiga en ambos segmentos
    def within(a, b, p):
        return min(a, b) - 1e-9 <= p <= max(a, b) + 1e-9

    if (
        within(x1, x2, px) and within(y1, y2, py) and
        within(x3, x4, px) and within(y3, y4, py)
    ):
        return np.array([px, py])

    return None

# cerrar polígono
closed = np.vstack([pts, pts[0]])

fig, ax = plt.subplots(figsize=(8, 8))

# polígono completo
ax.plot(closed[:, 0], closed[:, 1], marker="o")

# labels
for i, (x, y) in enumerate(pts, start=1):
    ax.text(x + 0.03, y + 0.03, f"C{i}", fontsize=10)

# edge 1 = C1->C2
e1_p1 = pts[0]
e1_p2 = pts[1]
ax.plot([e1_p1[0], e1_p2[0]], [e1_p1[1], e1_p2[1]], linewidth=3, label="Edge 1")

# edge 3 = C3->C4
e3_p1 = pts[2]
e3_p2 = pts[3]
ax.plot([e3_p1[0], e3_p2[0]], [e3_p1[1], e3_p2[1]], linewidth=3, label="Edge 3")

# intersección
inter = segment_intersection(e1_p1, e1_p2, e3_p1, e3_p2)
if inter is not None:
    ax.plot(inter[0], inter[1], marker="x", markersize=12, label="Intersection")
    print("Intersection:", inter)
else:
    print("No intersection found.")

ax.set_aspect("equal")
ax.grid(True)
ax.legend()
plt.show()

check_polygon("FLOOR", floor_pts)
check_polygon("CEILING", ceiling_pts)