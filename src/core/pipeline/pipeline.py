from core.evaluation.evaluator import Evaluator
from core.meshing.mesh_generator import Mesher
from core.simulation.simulator import Simulator


class Pipeline:
    def __init__(self):
        self.mesher = Mesher()
        self.simulator = Simulator()
        self.evaluator = Evaluator()

    def run(self, params):
        geo = self.geometry.generate(params)
        mesh = self.mesher.mesh(geo)
        sim = self.simulator.simulate(mesh)