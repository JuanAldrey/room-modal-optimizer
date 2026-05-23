from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator
from room_modal_optimizer.evaluation.modal_evaluator import Evaluator


class ModalPipeline:
    def __init__(self):
        self.mesher = Mesher()
        self.modalSimulator = ModalSimulator()
        self.modalEvaluator = Evaluator()

    def run(self, params, room_name='room'):
        mesh_path = self.mesher.create(params, lc=0.25, room_name=room_name, visualize=False)
        eig_freq, eig_vector, n_modes = self.modalSimulator.simulate(mesh_path, room_name=room_name, export=False)
        return self.modalEvaluator.evaluate(eig_freq, n_modes)