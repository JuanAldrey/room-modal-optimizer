class ModalPipeline:
    def __init__(self, mesher, simulator, evaluator):
        self.mesher = mesher
        self.modalSimulator = simulator
        self.modalEvaluator = evaluator
        self.failed_rooms = []

    def run(self, params, room_name='room', order=2, visualize=False, export=False):
        mesh_path = self.mesher.create(params, lc=0.25, room_name=room_name, visualize=visualize)
        if mesh_path is None:
            self.failed_rooms.append(params)
            return None
        eig_freq, eig_vector, n_modes = self.modalSimulator.simulate(mesh_path, room_name=room_name, order=order, export=export)
        return self.modalEvaluator.evaluate(eig_freq, n_modes)