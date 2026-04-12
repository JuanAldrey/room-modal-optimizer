# room-modal-optimizer
Tool for optimizing small-room geometry by evaluating modal response using genetic algorithms acoustic simulation tools like finite element methods.

The code is organized based on a genetic algorithm that runs the following pipeline as its fitness function:

1. Define the geometry of a given room
2. Discretize the geometry into finite elements
3. Simulate the room with finite elements methods
4. Evaluate the rooms performance

Each of the steps in the pipeline have their corresponding class:

class GeometryGenerator:
    def generate(self, params):
        input -> GA genes
        output -> .geo file

class Mesher:
    def mesh(self, geometry):
        input -> .geo file
        output -> .msh file 

class Simulator:
    def simulate(self, mesh):
        input -> .msh file
        output -> acoustic response

class Evaluator:
    def evaluate(self, simulation_result):
        input -> acoustic response
        output -> fitness (scalar value)

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
        return self.evaluator.evaluate(sim)

