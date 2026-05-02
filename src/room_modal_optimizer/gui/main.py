import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pyvista as pv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from . import dummy_functions as dumF

class RoomModalOptimizer(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=BOTH, expand=YES)

        # 1. Variables y Estilos
        self._init_vars()
        self._setup_styles()

        # 2. Layout Principal (75/25 aprox)
        self.columnconfigure(0, weight=3, uniform="main") # Panel Izquierdo
        self.columnconfigure(1, weight=1, uniform="main") # Panel Derecho
        self.rowconfigure(0, weight=1)

        self.left_panel = ttk.Frame(self)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.right_panel = ttk.Frame(self)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # 3. Construcción de Interfaz
        self.create_graphs_panel()
        self.create_right_functions()

    def _init_vars(self):
        """Inicializa el diccionario de parámetros y variables de control"""
        self.msh_path = ttk.StringVar()
        keys = [
            "Lx", "Ly", "Lz", "left_y0", "left_y1", "right_y0", "right_y1",
            "front_x0", "front_x1", "back_x0", "back_x1",
            "left_angle", "right_angle", "front_angle", "back_angle"
        ]
        self.room_params = {k: ttk.StringVar() for k in keys}

    def _setup_styles(self):
        """Configuración de fuentes globales"""
        self.style = ttk.Style()
        self.style.configure("Titulo.TLabel", font=("Helvetica", 11, "bold"))

    def create_graphs_panel(self):
        """Configura el área de visualización del mesh"""
        self.graph_frame = ttk.LabelFrame(self.left_panel, text=" Room Mesh View")
        self.graph_frame.pack(fill=BOTH, expand=YES, padx=5, pady=5)

        # Matplotlib Figure para embeber PyVista
        self.fig = plt.Figure(figsize=(5, 5), dpi=100, facecolor='#2b3e50')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=BOTH, expand=YES, padx=5, pady=5)

    def _create_input_group(self, parent, title, items, cols=3, uniform_tag="group"):
        """Función auxiliar para crear grupos de etiquetas y entradas de forma masiva"""
        ttk.Label(parent, text=title, style="Titulo.TLabel").pack(anchor="w", pady=(10, 5))
        
        container = ttk.Frame(parent)
        container.pack(fill=X, pady=(0, 10))
        
        # Configurar pesos de columnas para que los Entries sean elásticos
        entry_cols = [i*2 + 1 for i in range(cols)]
        container.columnconfigure(entry_cols, weight=1, uniform=uniform_tag)

        for i, (key, label) in enumerate(items):
            r, c = i // cols, (i % cols) * 2
            ttk.Label(container, text=label).grid(row=r, column=c, padx=2, pady=2, sticky="w")
            ttk.Entry(container, textvariable=self.room_params[key], width=5).grid(row=r, column=c+1, sticky="ew", padx=5, pady=2)

    def create_right_functions(self):
        """Organiza el panel de control derecho"""
        self.right_panel.columnconfigure(0, weight=1)
        
        # --- SECCIONES DE INPUT ---
        params_container = ttk.Frame(self.right_panel)
        params_container.pack(fill=BOTH, expand=YES)

        # 1. Lengths
        self._create_input_group(params_container, "Plant Lengths [m]:", 
                                [("Lx", "Lx:"), ("Ly", "Ly:"), ("Lz", "Lz:")], uniform_tag="len")
        
        # 2. Offsets (en 2 filas de 4 columnas para que quepan bien)
        off_items = [("left_y0", "L-Y0:"), ("left_y1", "L-Y1:"), ("right_y0", "R-Y0:"), ("right_y1", "R-Y1:"),
                     ("front_x0", "F-X0:"), ("front_x1", "F-X1:"), ("back_x0", "B-X0:"), ("back_x1", "B-X1:")]
        self._create_input_group(params_container, "Plant Offsets [m]:", off_items, cols=4, uniform_tag="off")

        # 3. Inclination
        ang_items = [("left_angle", "Left:"), ("right_angle", "Right:"), ("front_angle", "Front:"), ("back_angle", "Back:")]
        self._create_input_group(params_container, "Wall Inclination [deg]:", ang_items, cols=4, uniform_tag="ang")

        # --- BOTONES DE ACCIÓN ---
        btn_frame = ttk.Frame(self.right_panel)
        btn_frame.pack(fill=X, pady=20)
        btn_frame.columnconfigure((0, 1), weight=1, uniform="btns")

        # Configuración compacta de botones
        buttons = [
            ("Calculate", SUCCESS, self.execute_pipeline, 0, 0),
            ("Export Room Data", SECONDARY, self.export_rd, 0, 1),
            ("Export Modal Response", SECONDARY, self.export_mr, 1, 0),
            ("Clear", DANGER, self.clear_entries, 1, 1)
        ]

        for text, style, cmd, r, c in buttons:
            ttk.Button(btn_frame, text=text, bootstyle=style, command=cmd).grid(row=r, column=c, padx=5, pady=5, sticky="nsew")

    def display_mesh_pyvista(self, msh_path):
        """Renderizado de alta calidad con PyVista embebido"""
        if not msh_path: return
        
        try:
            mesh = pv.read(msh_path)
            plotter = pv.Plotter(off_screen=True)
            plotter.set_background("#2b3e50")
            
            plotter.add_mesh(mesh, show_edges=True, color="silver", edge_color="#1a1a1a", opacity=0.9)
            plotter.view_isometric()
            
            screenshot = plotter.screenshot()
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.imshow(screenshot)
            ax.axis('off')
            self.fig.tight_layout(pad=0)
            self.canvas.draw()
            plotter.close()

        except Exception as e:
            print(f"Error PyVista Render: {e}")

    # --- Callbacks ---
    def execute_pipeline(self): 
        # Recolectar datos
        data = {k: float(v.get()) for k, v in self.room_params.items()}      

        # Obtener path del dummy
        path = dumF.gui_get_mesh_path(data)
        if path:
            self.msh_path.set(path)
            self.display_mesh_pyvista(path)

    def export_rd(self): print("Exporting Room Data...")
    def export_mr(self): print("Exporting Modal Response...")
    def clear_entries(self):
        for var in self.room_params.values(): var.set("")
        self.fig.clear()
        self.canvas.draw()

if __name__ == "__main__":
    app = ttk.Window(title="Room Modal Optimizer", themename="superhero", size=(1400, 900))
    RoomModalOptimizer(app)
    app.mainloop()