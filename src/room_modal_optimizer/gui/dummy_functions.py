from src.room_modal_optimizer.meshing.mesher import Mesher
from src.room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from src.room_modal_optimizer.simulation.modal_simulator import ModalSimulator
from src.room_modal_optimizer.evaluation.modal_evaluator import Evaluator


def modal_sim(params):
    mesher = Mesher()
    modalSimulator = ModalSimulator()
    modalEvaluator = Evaluator()
    mesh_path = mesher.create(params, room_name="standard", visualize=False, source_pos=(2.5, 2.5, 1.5))
    freqs, _, n_modes = modalSimulator.simulate(mesh_path, export=False)
    print(n_modes)
    fsi = modalEvaluator.evaluate(freqs, n_modes)

    return freqs, fsi