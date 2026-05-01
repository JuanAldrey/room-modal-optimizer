import ttkbootstrap as ttk
from ttkbootstrap.constants import *

class RoomModalOptimizer(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=BOTH, expand=YES)

        self.room_params = {
            # Plant lengths
            "Lx": ttk.StringVar(), "Ly": ttk.StringVar(), "Lz": ttk.StringVar(),
        
            # Plant offsets
            "left_y0": ttk.StringVar(), "left_y1": ttk.StringVar(),
            "right_y0": ttk.StringVar(), "right_y1": ttk.StringVar(),
            "front_x0": ttk.StringVar(), "front_x1": ttk.StringVar(),
            "back_x0": ttk.StringVar(),  "back_x1": ttk.StringVar(),
        
            # Wall inclination (degrees)
            "left_angle": ttk.StringVar(), "right_angle": ttk.StringVar(),
            "front_angle": ttk.StringVar(), "back_angle": ttk.StringVar()
        }


        #Cuadrícula Principal
        self.columnconfigure(0, weight=80)
        self.columnconfigure(1, weight=20)
        self.rowconfigure(0, weight=1)

        # Panel Izquierdo (Gráficos)
        self.left_panel = ttk.Frame(self)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Panel Derecho (Parámetros y Funciones)
        self.right_panel = ttk.Frame(self)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.create_graphs()
        self.create_right_functions()

    def create_graphs(self):
        """Área reservada para los gráficos"""
       
        placeholder = ttk.LabelFrame(self.left_panel, text=" Gráficos y Resultados")
        placeholder.pack(fill=BOTH, expand=YES)
        


    def create_right_functions(self):
        self.right_panel.rowconfigure(0, weight=1)
        self.right_panel.rowconfigure(1, weight=0)
        self.right_panel.columnconfigure(0, weight=1)

        # Contenedor principal de parámetros (Scrollable si fuera necesario en el futuro)
        params_frame = ttk.Frame(self.right_panel)
        params_frame.grid(row=0, column=0, sticky="nsew")
        params_frame.columnconfigure(0, weight=1)

        # --- 1. PLANT LENGTHS ---
        ttk.Label(params_frame, text="Plant Lengths [m]:", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(10, 5))
        
        len_container = ttk.Frame(params_frame)
        len_container.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        len_container.columnconfigure((1, 3, 5), weight=1, uniform="len")
        
        for i, (key, label) in enumerate([("Lx", "Lx:"), ("Ly", "Ly:"), ("Lz", "Lz:")]):
            ttk.Label(len_container, text=label).grid(row=0, column=i*2, padx=2)
            ttk.Entry(len_container, textvariable=self.room_params[key], width=2).grid(row=0, column=i*2+1, sticky="ew", padx=5)

        # --- 2. PLANT OFFSETS ---
        ttk.Label(params_frame, text="Plant Offsets [m]:", font=("Helvetica", 12, "bold")).grid(row=2, column=0, sticky="w", pady=(10, 5))
        
        off_container = ttk.Frame(params_frame)
        off_container.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        off_container.columnconfigure((1, 3, 5, 7), weight=1, uniform="off")

        # Disposición en 2 filas para que no quede muy apretado
        off_items = [
            ("left_y0", "L-Y0:"), ("left_y1", "L-Y1:"), ("right_y0", "R-Y0:"), ("right_y1", "R-Y1:"),
            ("front_x0", "F-X0:"), ("front_x1", "F-X1:"), ("back_x0", "B-X0:"), ("back_x1", "B-X1:")
        ]
        
        for i, (key, label) in enumerate(off_items):
            row = i // 4
            col = (i % 4) * 2
            ttk.Label(off_container, text=label).grid(row=row, column=col, padx=2, pady=2)
            ttk.Entry(off_container, textvariable=self.room_params[key], width=2).grid(row=row, column=col+1, sticky="ew", padx=5, pady=2)

        # --- 3. WALL INCLINATION ---
        ttk.Label(params_frame, text="Wall Inclination [deg]:", font=("Helvetica", 12, "bold")).grid(row=4, column=0, sticky="w", pady=(10, 5))
        
        ang_container = ttk.Frame(params_frame)
        ang_container.grid(row=5, column=0, sticky="ew", pady=(0, 15))
        ang_container.columnconfigure((1, 3, 5, 7), weight=1, uniform="ang")

        ang_items = [
            ("left_angle", "Left:"), ("right_angle", "Right:"), 
            ("front_angle", "Front:"), ("back_angle", "Back:")
        ]

        for i, (key, label) in enumerate(ang_items):
            ttk.Label(ang_container, text=label).grid(row=0, column=i*2, padx=2)
            ttk.Entry(ang_container, textvariable=self.room_params[key], width=2).grid(row=0, column=i*2+1, sticky="ew", padx=5)


        # Contenedor de funciones
        functions_frame = ttk.Frame(self.right_panel)
        functions_frame.grid(row=1, column=0, sticky="nsew", pady=15)

        functions_frame.columnconfigure(0, weight=1) # Espacio izquierdo
        functions_frame.columnconfigure(1, weight=2) # Botón col 1
        functions_frame.columnconfigure(2, weight=2) # Botón col 2
        functions_frame.columnconfigure(3, weight=1) # Espacio derecho
        
        functions_frame.rowconfigure(0, weight=1)
        functions_frame.rowconfigure(1, weight=1)

        self.btn_calc = ttk.Button(
            functions_frame, text="Calculate", bootstyle=SUCCESS, command=self.execute_pipeline
        )
        self.btn_calc.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.btn_rd = ttk.Button(
            functions_frame, text="Export Room Data", bootstyle=SECONDARY, command=self.export_rd
        )
        self.btn_rd.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        self.btn_mr = ttk.Button(
            functions_frame, text="Export Modal Response", bootstyle=SECONDARY, command=self.export_mr
        )
        self.btn_mr.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.btn_clear = ttk.Button(
            functions_frame, text="Clear", bootstyle=DANGER, command=self.clear_entries
        )
        self.btn_clear.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

    # --- Callbacks ---
    def execute_pipeline(self): print("Calculate")
    def export_rd(self): print("Export RD")
    def export_mr(self): print("Export MR")
    def clear_entries(self): print("Clear")

if __name__ == "__main__":
    app = ttk.Window(
        title="Room Modal Optimizer", 
        themename="superhero", 
        size=(1400, 900)
    )
    RoomModalOptimizer(app)
    app.mainloop()