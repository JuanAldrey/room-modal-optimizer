import math
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


# ============================================================
# 1. DIBUJO 2D DE LA PLANTA
# ============================================================

def draw_floor_polygon():
    """
    Editor simple de planta con Matplotlib.

    Click izquierdo: agregar vértice
    Botón 'Borrar último': elimina el último vértice
    Botón 'Terminar': cierra la ventana y continúa

    Esta versión usa plt.pause() para funcionar mejor en Spyder.
    """
    from matplotlib.widgets import Button

    vertices = []
    finished = {"value": False}

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.set_title(
        "Dibujar planta del recinto\n"
        "Click izquierdo: agregar vértice | Terminar: continuar"
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 6)

    point_plot, = ax.plot([], [], "o", markersize=6)
    line_plot, = ax.plot([], [], "-", linewidth=1.5)

    text_labels = []

    def update_plot():
        nonlocal text_labels

        for label in text_labels:
            label.remove()
        text_labels = []

        if len(vertices) == 0:
            point_plot.set_data([], [])
            line_plot.set_data([], [])
            fig.canvas.draw_idle()
            return

        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]

        point_plot.set_data(xs, ys)

        # Dibujar cierre provisional del polígono
        if len(vertices) > 1:
            line_plot.set_data(xs + [xs[0]], ys + [ys[0]])
        else:
            line_plot.set_data(xs, ys)

        for i, (x, y) in enumerate(vertices):
            label = ax.text(x, y, f" V{i+1}", fontsize=9)
            text_labels.append(label)

        fig.canvas.draw_idle()

    def on_click(event):
        # Solo agregar puntos si el click fue dentro del eje principal
        if event.inaxes != ax:
            return

        if event.button == 1:
            vertices.append((float(event.xdata), float(event.ydata)))
            update_plot()

    def finish(event):
        if len(vertices) < 3:
            print("Se necesitan al menos 3 vértices para formar un área.")
            return

        finished["value"] = True
        plt.close(fig)

    def undo(event):
        if vertices:
            vertices.pop()
            update_plot()

    # Botones
    ax_undo = fig.add_axes([0.45, 0.06, 0.20, 0.08])
    ax_finish = fig.add_axes([0.70, 0.06, 0.20, 0.08])

    btn_undo = Button(ax_undo, "Borrar último")
    btn_finish = Button(ax_finish, "Terminar")

    btn_undo.on_clicked(undo)
    btn_finish.on_clicked(finish)

    # Guardar referencias para que Spyder no pierda los botones
    fig._btn_undo = btn_undo
    fig._btn_finish = btn_finish

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show(block=False)

    # Loop manual: suele funcionar mejor en Spyder
    while plt.fignum_exists(fig.number) and not finished["value"]:
        plt.pause(0.05)

    print("\nDibujo finalizado.")
    print("Vértices capturados:", vertices)

    return vertices


# ============================================================
# 2. FUNCIONES GEOMÉTRICAS
# ============================================================

def polygon_signed_area(vertices):
    """
    Devuelve el área firmada del polígono.
    Si es positiva, los vértices están en sentido antihorario.
    """
    area = 0.0
    n = len(vertices)

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return 0.5 * area


def polygon_area(vertices):
    return abs(polygon_signed_area(vertices))


def ensure_counterclockwise(vertices):
    """
    Asegura que los vértices estén en sentido antihorario.
    """
    if polygon_signed_area(vertices) < 0:
        return list(reversed(vertices))
    return vertices


def inward_normal(v1, v2):
    """
    Calcula la normal interna de una pared.
    Asume polígono en sentido antihorario.

    Para un polígono antihorario, el interior queda a la izquierda
    de cada segmento.
    """
    x1, y1 = v1
    x2, y2 = v2

    edge = np.array([x2 - x1, y2 - y1], dtype=float)
    length = np.linalg.norm(edge)

    if length < 1e-9:
        raise ValueError("Hay dos vértices consecutivos iguales.")

    edge /= length

    # Normal izquierda: apunta hacia el interior si el polígono es antihorario
    normal = np.array([-edge[1], edge[0]])

    return normal


def offset_line(v1, v2, offset):
    """
    Devuelve una recta desplazada paralelamente a la pared V1-V2.

    Recta:
        p + t*d

    offset positivo desplaza hacia adentro.
    """
    p1 = np.array(v1, dtype=float)
    p2 = np.array(v2, dtype=float)

    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1e-9:
        raise ValueError("Hay una pared de longitud cero.")

    direction /= length
    normal = inward_normal(v1, v2)

    shifted_p = p1 + offset * normal

    return shifted_p, direction


def line_intersection(p1, d1, p2, d2):
    """
    Intersección entre dos rectas 2D:
        L1 = p1 + t*d1
        L2 = p2 + u*d2
    """
    A = np.array([
        [d1[0], -d2[0]],
        [d1[1], -d2[1]]
    ])

    b = p2 - p1
    det = np.linalg.det(A)

    if abs(det) < 1e-9:
        raise ValueError(
            "Dos paredes desplazadas quedaron paralelas o casi paralelas. "
            "Probá reducir inclinaciones o modificar la planta."
        )

    t, _ = np.linalg.solve(A, b)
    return p1 + t * d1


def compute_ceiling_vertices(floor_vertices, height, wall_tilts_deg):
    """
    Calcula los vértices superiores del recinto.

    Cada pared se desplaza:
        offset = height * tan(theta)

    Convención:
        tilt positivo  -> pared se inclina hacia adentro
        tilt negativo  -> pared se inclina hacia afuera

    Los vértices superiores se obtienen como intersección
    de las líneas desplazadas adyacentes.
    """
    if len(floor_vertices) != len(wall_tilts_deg):
        raise ValueError("Debe haber una inclinación por cada pared.")

    floor_vertices = ensure_counterclockwise(floor_vertices)

    n = len(floor_vertices)
    shifted_lines = []

    for i in range(n):
        v1 = floor_vertices[i]
        v2 = floor_vertices[(i + 1) % n]

        theta = math.radians(wall_tilts_deg[i])
        offset = height * math.tan(theta)

        shifted_lines.append(offset_line(v1, v2, offset))

    ceiling_vertices = []

    for i in range(n):
        # El vértice superior Ci se obtiene como intersección entre:
        # pared anterior desplazada y pared actual desplazada.
        prev_line = shifted_lines[(i - 1) % n]
        curr_line = shifted_lines[i]

        p_prev, d_prev = prev_line
        p_curr, d_curr = curr_line

        c = line_intersection(p_prev, d_prev, p_curr, d_curr)
        ceiling_vertices.append((float(c[0]), float(c[1]), float(height)))

    return floor_vertices, ceiling_vertices


# ============================================================
# 3. VALIDACIÓN SIMPLE
# ============================================================

def check_basic_geometry(floor_vertices, ceiling_vertices):
    """
    Validaciones básicas.
    Más adelante conviene agregar shapely para detectar autointersecciones.
    """
    if len(floor_vertices) < 3:
        raise ValueError("La planta debe tener al menos 3 vértices.")

    area_floor = polygon_area(floor_vertices)
    area_ceiling = polygon_area([(x, y) for x, y, _ in ceiling_vertices])

    if area_floor < 1e-6:
        raise ValueError("El área del piso es demasiado chica o inválida.")

    if area_ceiling < 1e-6:
        raise ValueError("El área del techo es demasiado chica o inválida.")

    print(f"\nÁrea piso aproximada:   {area_floor:.3f} m²")
    print(f"Área techo aproximada: {area_ceiling:.3f} m²")


# ============================================================
# 4. CREAR MALLA SUPERFICIAL PARA PYVISTA
# ============================================================

def build_pyvista_room(floor_vertices_2d, ceiling_vertices_3d):
    """
    Construye una superficie cerrada para visualizar.

    No es la malla FEM todavía.
    Es una preview geométrica del recinto.
    """
    n = len(floor_vertices_2d)

    floor_vertices_3d = [
        (x, y, 0.0) for x, y in floor_vertices_2d
    ]

    points = np.array(floor_vertices_3d + ceiling_vertices_3d)

    faces = []

    # Piso: invertido para que la normal apunte hacia abajo
    faces.append([n] + list(reversed(range(n))))

    # Techo
    faces.append([n] + list(range(n, 2 * n)))

    # Paredes laterales
    for i in range(n):
        j = (i + 1) % n
        faces.append([4, i, j, j + n, i + n])

    faces_flat = np.hstack(faces)
    mesh = pv.PolyData(points, faces_flat)

    return mesh


# ============================================================
# 5. INPUT DE ALTURA E INCLINACIONES
# ============================================================

def ask_height():
    while True:
        try:
            height = float(input("\nAltura del recinto [m]: "))
            if height <= 0:
                print("La altura debe ser mayor a 0.")
                continue
            return height
        except ValueError:
            print("Ingresá un número válido.")


def ask_wall_tilts(vertices):
    """
    Pide inclinación de cada pared.

    Convención:
        positivo = hacia adentro
        negativo = hacia afuera
    """
    tilts = []
    n = len(vertices)

    print("\nDefinir inclinación vertical de cada pared.")
    print("Convención: positivo = hacia adentro, negativo = hacia afuera.")
    print("Ejemplo: 5 significa que la parte superior de la pared se mete hacia adentro 5 grados.")
    print("Si dejás vacío, se usa 0°.\n")

    for i in range(n):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]

        wall_name = f"V{i+1}-V{(i + 2) if (i + 1) < n else 1}"

        while True:
            try:
                value = input(
                    f"Inclinación pared {wall_name} "
                    f"({v1} -> {v2}) [deg, default 0]: "
                )

                if value.strip() == "":
                    tilt = 0.0
                else:
                    tilt = float(value)

                if abs(tilt) > 20:
                    print("Advertencia: ángulos mayores a ±20° pueden generar geometrías raras.")
                    confirm = input("¿Usar igual? [s/N]: ").strip().lower()
                    if confirm != "s":
                        continue

                tilts.append(tilt)
                break

            except ValueError:
                print("Ingresá un número válido.")

    return tilts


# ============================================================
# 6. VISUALIZACIÓN 3D
# ============================================================

def visualize_room(floor_vertices_2d, ceiling_vertices_3d):
    room_mesh = build_pyvista_room(floor_vertices_2d, ceiling_vertices_3d)

    floor_points = np.array([(x, y, 0.0) for x, y in floor_vertices_2d])
    ceiling_points = np.array(ceiling_vertices_3d)

    plotter = pv.Plotter()
    plotter.add_mesh(
        room_mesh,
        show_edges=True,
        opacity=0.45,
    )

    plotter.add_points(
        floor_points,
        point_size=12,
        render_points_as_spheres=True,
    )

    plotter.add_points(
        ceiling_points,
        point_size=12,
        render_points_as_spheres=True,
    )

    # Etiquetas de vértices inferiores
    for i, p in enumerate(floor_points):
        plotter.add_point_labels(
            [p],
            [f"V{i+1}"],
            font_size=12,
            point_size=0,
        )

    # Etiquetas de vértices superiores
    for i, p in enumerate(ceiling_points):
        plotter.add_point_labels(
            [p],
            [f"C{i+1}"],
            font_size=12,
            point_size=0,
        )

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()


# ============================================================
# 7. EXPORTAR DATOS PARA GMSH / BACKEND
# ============================================================

def build_geometry_dict(floor_vertices_2d, ceiling_vertices_3d, wall_tilts_deg):
    """
    Devuelve una estructura que luego pueden usar para Gmsh.

    floor_vertices:
        V1, V2, ..., Vn en z=0

    ceiling_vertices:
        C1, C2, ..., Cn en z=height

    walls:
        W1 conecta V1-V2-C2-C1
        W2 conecta V2-V3-C3-C2
        ...
    """
    room = {
        "floor_vertices": [],
        "ceiling_vertices": [],
        "walls": []
    }

    n = len(floor_vertices_2d)

    for i, (x, y) in enumerate(floor_vertices_2d):
        room["floor_vertices"].append({
            "id": f"V{i+1}",
            "x": float(x),
            "y": float(y),
            "z": 0.0
        })

    for i, (x, y, z) in enumerate(ceiling_vertices_3d):
        room["ceiling_vertices"].append({
            "id": f"C{i+1}",
            "x": float(x),
            "y": float(y),
            "z": float(z)
        })

    for i in range(n):
        j = (i + 1) % n
        room["walls"].append({
            "id": f"W{i+1}",
            "floor_from": f"V{i+1}",
            "floor_to": f"V{j+1}",
            "ceiling_from": f"C{i+1}",
            "ceiling_to": f"C{j+1}",
            "tilt_deg": float(wall_tilts_deg[i])
        })

    return room


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    print("\n=== EDITOR SIMPLE DE PLANTA PARA RECINTO ===")
    print("Se abrirá una ventana para dibujar el área del piso.")
    print("En Spyder, asegurate de haber ejecutado antes: %matplotlib qt\n")

    floor_vertices = draw_floor_polygon()

    print("\nVolví de la ventana de dibujo.")
    print("Cantidad de vértices:", len(floor_vertices))

    if len(floor_vertices) < 3:
        raise RuntimeError("No se dibujó una planta válida.")

    # Asegurar sentido antihorario
    floor_vertices = ensure_counterclockwise(floor_vertices)

    print("\nVértices dibujados:")
    for i, v in enumerate(floor_vertices, start=1):
        print(f"V{i}: x={v[0]:.3f}, y={v[1]:.3f}")

    height = ask_height()
    wall_tilts = ask_wall_tilts(floor_vertices)

    floor_vertices, ceiling_vertices = compute_ceiling_vertices(
        floor_vertices,
        height,
        wall_tilts
    )

    check_basic_geometry(floor_vertices, ceiling_vertices)

    print("\nVÉRTICES DE TECHO CALCULADOS")
    for i, v in enumerate(ceiling_vertices, start=1):
        print(f"C{i}: x={v[0]:.3f}, y={v[1]:.3f}, z={v[2]:.3f}")

    geometry_data = build_geometry_dict(
        floor_vertices,
        ceiling_vertices,
        wall_tilts
    )

    print("\nEstructura generada para pasar luego a Gmsh:")
    print(geometry_data)

    visualize_room(floor_vertices, ceiling_vertices)