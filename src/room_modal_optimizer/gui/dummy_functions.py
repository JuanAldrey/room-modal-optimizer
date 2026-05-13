from src.room_modal_optimizer.meshing.mesher import Mesher
from src.room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from src.room_modal_optimizer.simulation.modal_simulator import ModalSimulator

def gui_get_mesh_path(params):
    print(params)
    mesher = Mesher()
    mesh_path = mesher.create(params, room_name="standard", visualize=False, source_pos=(2.5, 2.5, 1.5))
    return mesh_path

def gui_modal_dist(mesh_path):
    modalSimulator = ModalSimulator()
    modalSimulator.simulate(mesh_path, export=True)
    #falta obtener los modos