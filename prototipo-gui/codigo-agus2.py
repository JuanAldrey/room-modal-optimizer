# -- coding: utf-8 --
"""
Created on Sat May 16 21:47:54 2026

@author: Agust
"""

# -- coding: utf-8 --
"""
Editor simple de planta + selector de paredes + configuración por popup + preview 3D

Recomendado en Spyder:
    %matplotlib qt

Dependencias:
    pip install numpy matplotlib pyvista
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

import tkinter as tk
from tkinter import simpledialog, messagebox


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

        if len(vertices) > 1:
            line_plot.set_data(xs + [xs[0]], ys + [ys[0]])
        else:
            line_plot.set_data(xs, ys)

        for i, (x, y) in enumerate(vertices):
            label = ax.text(x, y, f" V{i+1}", fontsize=9)
            text_labels.append(label)

        fig.canvas.draw_idle()

    def on_click(event):
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

    # Normal izquierda: apunta hacia el interior si el polígono es antihorario.
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
        prev_line = shifted_lines[(i - 1) % n]
        curr_line = shifted_lines[i]

        p_prev, d_prev = prev_line
        p_curr, d_curr = curr_line

        c = line_intersection(p_prev, d_prev, p_curr, d_curr)
        ceiling_vertices.append((float(c[0]), float(c[1]), float(height)))

    return floor_vertices, ceiling_vertices


# ============================================================
# 3. SELECCIÓN DE PAREDES TIPO "VARITA" EN 2D
# ============================================================

def point_to_segment_distance(point, a, b):
    """
    Distancia entre un punto y un segmento 2D.
    """
    p = np.array(point, dtype=float)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    ab = b - a
    ab_len2 = np.dot(ab, ab)

    if ab_len2 < 1e-12:
        return np.linalg.norm(p - a)

    t = np.dot(p - a, ab) / ab_len2
    t = max(0.0, min(1.0, t))

    projection = a + t * ab
    return np.linalg.norm(p - projection)


def find_nearest_wall(click_point, floor_vertices, tolerance=0.35):
    """
    Devuelve el índice de la pared más cercana al click.
    Si ninguna está dentro de la tolerancia, devuelve None.
    """
    n = len(floor_vertices)
    distances = []

    for i in range(n):
        a = floor_vertices[i]
        b = floor_vertices[(i + 1) % n]
        d = point_to_segment_distance(click_point, a, b)
        distances.append(d)

    idx = int(np.argmin(distances))

    if distances[idx] <= tolerance:
        return idx

    return None


def create_default_wall_props(floor_vertices):
    """
    Crea propiedades por defecto para cada pared.
    """
    n = len(floor_vertices)
    walls = []

    for i in range(n):
        j = (i + 1) % n
        walls.append({
            "id": f"W{i+1}",
            "from": f"V{i+1}",
            "to": f"V{j+1}",
            "tilt_deg": 0.0,
            "locked": False,
            "optimize_tilt": False,
            "tilt_min": 0.0,
            "tilt_max": 0.0
        })

    return walls


def print_wall_table(wall_props):
    """
    Imprime una tabla simple de paredes en consola.
    """
    print("\n=== PROPIEDADES DE PAREDES ===")
    for wall in wall_props:
        print(
            f"{wall['id']}: {wall['from']}->{wall['to']} | "
            f"tilt={wall['tilt_deg']}° | "
            f"locked={wall['locked']} | "
            f"opt_tilt={wall['optimize_tilt']} | "
            f"range=[{wall['tilt_min']}, {wall['tilt_max']}]"
        )


def configure_wall_dialog(wall):
    """
    Configura una pared seleccionada usando ventanas emergentes.
    Evita depender de input() en la consola de Spyder.
    """
    root = tk.Tk()
    root.withdraw()

    try:
        # Inclinación actual
        tilt = simpledialog.askfloat(
            "Configurar pared",
            f"{wall['id']} ({wall['from']} -> {wall['to']})\n\n"
            f"Inclinación actual [deg]:",
            initialvalue=wall["tilt_deg"],
            parent=root
        )

        if tilt is None:
            root.destroy()
            return wall

        wall["tilt_deg"] = float(tilt)

        # Bloqueo
        locked_answer = messagebox.askyesno(
            "Bloqueo de pared",
            f"¿Querés bloquear {wall['id']}?\n\n"
            "Sí = no se modificará más adelante\n"
            "No = queda disponible",
            parent=root
        )

        wall["locked"] = bool(locked_answer)

        # Optimización futura GA
        optimize_answer = messagebox.askyesno(
            "Optimización GA",
            f"¿Querés permitir que el GA optimice la inclinación de {wall['id']}?",
            parent=root
        )

        wall["optimize_tilt"] = bool(optimize_answer)

        if wall["optimize_tilt"]:
            tilt_min = simpledialog.askfloat(
                "Rango de inclinación",
                f"{wall['id']} - inclinación mínima [deg]:",
                initialvalue=wall.get("tilt_min", wall["tilt_deg"]),
                parent=root
            )

            if tilt_min is None:
                tilt_min = wall["tilt_deg"]

            tilt_max = simpledialog.askfloat(
                "Rango de inclinación",
                f"{wall['id']} - inclinación máxima [deg]:",
                initialvalue=wall.get("tilt_max", wall["tilt_deg"]),
                parent=root
            )

            if tilt_max is None:
                tilt_max = wall["tilt_deg"]

            if tilt_min > tilt_max:
                messagebox.showwarning(
                    "Rango inválido",
                    "El mínimo era mayor que el máximo. Se intercambiaron los valores.",
                    parent=root
                )
                tilt_min, tilt_max = tilt_max, tilt_min

            wall["tilt_min"] = float(tilt_min)
            wall["tilt_max"] = float(tilt_max)

        else:
            wall["tilt_min"] = wall["tilt_deg"]
            wall["tilt_max"] = wall["tilt_deg"]

        messagebox.showinfo(
            "Pared actualizada",
            f"{wall['id']} actualizada:\n\n"
            f"Inclinación: {wall['tilt_deg']}°\n"
            f"Bloqueada: {wall['locked']}\n"
            f"Optimizar inclinación: {wall['optimize_tilt']}\n"
            f"Rango: [{wall['tilt_min']}, {wall['tilt_max']}]",
            parent=root
        )

    finally:
        try:
            root.destroy()
        except Exception:
            pass

    return wall


def wall_selection_window(floor_vertices, wall_props):
    """
    Ventana para seleccionar paredes en 2D.

    Click cerca de una pared: selecciona pared
    Botón Configurar pared: cierra ventana y devuelve índice seleccionado
    Botón Terminar configuración: cierra ventana y termina configuración

    Devuelve:
        ("configure", selected_idx)
        ("finish", None)
    """
    from matplotlib.widgets import Button

    action = {"type": None}
    selected_idx = {"value": None}

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.set_title(
        "Seleccionar pared\n"
        "Click cerca de una pared | Configurar pared | Terminar configuración"
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    xs = [v[0] for v in floor_vertices]
    ys = [v[1] for v in floor_vertices]

    margin = 1.0
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    # Polígono base
    ax.plot(xs + [xs[0]], ys + [ys[0]], "-", linewidth=1.5)
    ax.plot(xs, ys, "o", markersize=6)

    # Etiquetas vértices
    for i, (x, y) in enumerate(floor_vertices):
        ax.text(x, y, f" V{i+1}", fontsize=9)

    # Etiquetas paredes
    n = len(floor_vertices)
    for i in range(n):
        a = np.array(floor_vertices[i])
        b = np.array(floor_vertices[(i + 1) % n])
        mid = 0.5 * (a + b)
        ax.text(mid[0], mid[1], f" {wall_props[i]['id']}", fontsize=10)

    selected_plot, = ax.plot([], [], linewidth=5)

    info_text = ax.text(
        0.02, 0.98,
        "Pared seleccionada: ninguna",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    def update_selected_wall(idx):
        if idx is None:
            selected_plot.set_data([], [])
            info_text.set_text("Pared seleccionada: ninguna")
            fig.canvas.draw_idle()
            return

        i = idx
        j = (i + 1) % len(floor_vertices)

        x1, y1 = floor_vertices[i]
        x2, y2 = floor_vertices[j]

        selected_plot.set_data([x1, x2], [y1, y2])

        wall = wall_props[i]
        info_text.set_text(
            f"Pared seleccionada: {wall['id']}\n"
            f"{wall['from']}->{wall['to']}\n"
            f"tilt={wall['tilt_deg']}° | locked={wall['locked']}\n"
            f"opt_tilt={wall['optimize_tilt']} | "
            f"range=[{wall['tilt_min']}, {wall['tilt_max']}]"
        )

        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return

        idx = find_nearest_wall(
            click_point=(event.xdata, event.ydata),
            floor_vertices=floor_vertices,
            tolerance=0.35
        )

        if idx is not None:
            selected_idx["value"] = idx
            update_selected_wall(idx)

    def configure_selected(event):
        if selected_idx["value"] is None:
            print("Primero seleccioná una pared.")
            return

        action["type"] = "configure"
        plt.close(fig)

    def finish(event):
        action["type"] = "finish"
        plt.close(fig)

    ax_configure = fig.add_axes([0.42, 0.06, 0.24, 0.08])
    ax_finish = fig.add_axes([0.70, 0.06, 0.24, 0.08])

    btn_configure = Button(ax_configure, "Configurar pared")
    btn_finish = Button(ax_finish, "Terminar config.")

    btn_configure.on_clicked(configure_selected)
    btn_finish.on_clicked(finish)

    fig._btn_configure = btn_configure
    fig._btn_finish = btn_finish

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show(block=False)

    while plt.fignum_exists(fig.number) and action["type"] is None:
        plt.pause(0.05)

    if action["type"] == "configure":
        return "configure", selected_idx["value"]

    return "finish", None


def configure_walls_loop(floor_vertices, wall_props):
    """
    Loop de configuración de paredes.
    Permite seleccionar una pared, configurarla con popup,
    y volver a la ventana de selección.
    """
    while True:
        print_wall_table(wall_props)

        action, idx = wall_selection_window(floor_vertices, wall_props)

        if action == "finish":
            print("\nConfiguración de paredes finalizada.")
            break

        if action == "configure" and idx is not None:
            wall_props[idx] = configure_wall_dialog(wall_props[idx])

    return wall_props


# ============================================================
# 4. VALIDACIÓN SIMPLE
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
# 5. CREAR MALLA SUPERFICIAL PARA PYVISTA
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
# 6. INPUT DE ALTURA
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


# ============================================================
# 7. VISUALIZACIÓN 3D
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

    for i, p in enumerate(floor_points):
        plotter.add_point_labels(
            [p],
            [f"V{i+1}"],
            font_size=12,
            point_size=0,
        )

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
# 8. EXPORTAR DATOS PARA GMSH / BACKEND
# ============================================================

def build_geometry_dict(floor_vertices_2d, ceiling_vertices_3d, wall_props):
    """
    Devuelve una estructura que luego pueden usar para Gmsh.

    Incluye:
    - vértices inferiores
    - vértices superiores
    - paredes/caras laterales
    - propiedades de cada pared
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
        wall = wall_props[i]

        room["walls"].append({
            "id": wall["id"],
            "floor_from": f"V{i+1}",
            "floor_to": f"V{j+1}",
            "ceiling_from": f"C{i+1}",
            "ceiling_to": f"C{j+1}",

            # Propiedades útiles para GUI / futuro GA
            "tilt_deg": float(wall["tilt_deg"]),
            "locked": bool(wall["locked"]),
            "optimize_tilt": bool(wall["optimize_tilt"]),
            "tilt_min": float(wall["tilt_min"]),
            "tilt_max": float(wall["tilt_max"]),
        })

    return room


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":
    print("\n=== EDITOR SIMPLE DE PLANTA PARA RECINTO ===")
    print("Se abrirá una ventana para dibujar el área del piso.")
    print("En Spyder, asegurate de haber ejecutado antes: %matplotlib qt\n")

    # 1. Dibujar planta
    floor_vertices = draw_floor_polygon()

    print("\nVolví de la ventana de dibujo.")
    print("Cantidad de vértices:", len(floor_vertices))

    if len(floor_vertices) < 3:
        raise RuntimeError("No se dibujó una planta válida.")

    # 2. Asegurar sentido antihorario
    floor_vertices = ensure_counterclockwise(floor_vertices)

    print("\nVértices dibujados:")
    for i, v in enumerate(floor_vertices, start=1):
        print(f"V{i}: x={v[0]:.3f}, y={v[1]:.3f}")

    # 3. Pedir altura
    height = ask_height()

    # 4. Crear propiedades de paredes
    wall_props = create_default_wall_props(floor_vertices)

    # 5. Herramienta tipo varita para configurar paredes
    wall_props = configure_walls_loop(floor_vertices, wall_props)

    # 6. Extraer inclinaciones
    wall_tilts = [wall["tilt_deg"] for wall in wall_props]

    # 7. Calcular techo
    floor_vertices, ceiling_vertices = compute_ceiling_vertices(
        floor_vertices,
        height,
        wall_tilts
    )

    # 8. Validar geometría
    check_basic_geometry(floor_vertices, ceiling_vertices)

    print("\nVÉRTICES DE TECHO CALCULADOS")
    for i, v in enumerate(ceiling_vertices, start=1):
        print(f"C{i}: x={v[0]:.3f}, y={v[1]:.3f}, z={v[2]:.3f}")

    # 9. Construir estructura para Gmsh / backend
    geometry_data = build_geometry_dict(
        floor_vertices,
        ceiling_vertices,
        wall_props
    )

    print("\nEstructura generada para pasar luego a Gmsh:")
    print(geometry_data)

    # 10. Visualización 3D
    visualize_room(floor_vertices, ceiling_vertices)