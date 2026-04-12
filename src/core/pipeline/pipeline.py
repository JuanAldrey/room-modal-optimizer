from core.evaluation.evaluator import Evaluator
from core.geometry.geometry_generator import GeometryGenerator
from core.meshing.mesher import Mesher
from core.simulation.simulator import Simulator


class Pipeline:
    def __init__(self):
        self.geometry = GeometryGenerator()
        self.mesher = Mesher()
        self.simulator = Simulator()
        self.evaluator = Evaluator()

    def run(self, params):
        geo = self.geometry.generate(params)
        mesh = self.mesher.mesh(geo)
        sim = self.simulator.simulate(mesh)