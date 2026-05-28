import copy
import numpy as np
import pygad

from room_modal_optimizer.pipeline.modal_pipeline import ModalPipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator
from room_modal_optimizer.evaluation.modal_evaluator import ModalEvaluator

class GeneticAlgorithm:
    def __init__(self, gene_space_config):
        # TODO: define dynamic params
        self.BASE_PARAMS = {
            "data": {
                "vertices": {
                    "V1": [0.0, 0.0],
                    "V2": [0.0, 0.0],
                    "V3": [0.0, 0.0],
                    "V4": [0.0, 0.0],
                    "V5": [0.0, 0.0],
                    "V6": [0.0, 0.0],
                    "V7": [0.0, 0.0],
                    "V8": [0.0, 0.0],
                },
                "walls": {
                    "W1": 0.0,
                    "W2": 0.0,
                    "W3": 0.0,
                    "W4": 0.0,
                    "W5": 0.0,
                    "W6": 0.0,
                    "W7": 0.0,
                    "W8": 0.0,
                },
                "Z": 0.0,
            }
        }

        self.mesher = Mesher()
        self.modal_simulator = ModalSimulator()
        self.modal_evaluator = ModalEvaluator()
        self.modal_pipeline = ModalPipeline(
            self.mesher,
            self.modal_simulator,
            self.modal_evaluator,
        )

        self.fitness_history = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "worst_fitness": [],
        }

        self.gene_space = self.define_gene_space(gene_space_config)
        self.ga_instance = None

    def run(self):
        self.ga_instance = self.create_ga_instance()
        self.ga_instance.run()

    # TODO: define dynamic param assignation
    def solution_to_params(self, solution):
        params = copy.deepcopy(self.BASE_PARAMS)

        # Genes 0 a 7: inclinaciones de paredes
        for i in range(8):
            params["data"]["walls"][f"W{i + 1}"] = float(solution[i])

        # Gen 8: altura
        params["data"]["Z"] = float(solution[8])
        
        #print(params)

        return params
    
    def fitness_func(self, ga_instance, solution, solution_idx):
        generation = ga_instance.generations_completed
        room_name = f"ga_modal_gen_{generation:03d}_sol_{solution_idx:03d}"

        params = self.solution_to_params(solution)

        idx = self.modal_pipeline.run(
            params,
            room_name=room_name,
            order=1,
            visualize=False,
            export=False,
        )
        
        if idx is None:
            fitness = 1e-9
        else:
            fitness = 1.0 / (1.0 + abs(idx))

        #print(f"{room_name} | idx={idx:.6f} | fitness={fitness:.6f}")

        return fitness
    
    def on_generation(self, ga_instance):
        generation = ga_instance.generations_completed
        fitness_values = ga_instance.last_generation_fitness

        best_fitness = np.max(fitness_values)
        mean_fitness = np.mean(fitness_values)
        worst_fitness = np.min(fitness_values)

        self.fitness_history["generation"].append(generation)
        self.fitness_history["best_fitness"].append(float(best_fitness))
        self.fitness_history["mean_fitness"].append(float(mean_fitness))
        self.fitness_history["worst_fitness"].append(float(worst_fitness))

        print("\n==============================")
        print("Generation:", generation)
        print("Best fitness:", best_fitness)
        print("Mean fitness:", mean_fitness)
        print("Worst fitness:", worst_fitness)
        print("==============================\n")

    # TODO: define a dynamic gene space
    def define_gene_space(self, gene_space_config):
        return [
            {"low": -5.0, "high": 5.0},  # W1
            {"low": -5.0, "high": 5.0},  # W2
            {"low": -5.0, "high": 5.0},  # W3
            {"low": -5.0, "high": 5.0},  # W4
            {"low": -5.0, "high": 5.0},  # W5
            {"low": -5.0, "high": 5.0},  # W6
            {"low": -5.0, "high": 5.0},  # W7
            {"low": -5.0, "high": 5.0},  # W8
            {"low": 2.4, "high": 3.6},   # Z
        ]
    
    # TODO: investigate better GA params
    def create_ga_instance(self):
        self.ga_instance = pygad.GA(
            num_generations=100,
            sol_per_pop=10,
            num_parents_mating=5,

            num_genes=len(self.gene_space),
            gene_space=self.gene_space,

            fitness_func=self.fitness_func,

            #parent_selection_type=,    -> default selection type
            #"K_tournament=3,

            crossover_type="single_point",

            mutation_type="random",
            mutation_probability=0.15,

            keep_elitism=3,

            on_generation=self.on_generation,

            random_seed=42,
        )

    def get_history(self):
        best_solution, best_fitness, best_idx = self.ga_instance.best_solution()
        best_params = self.solution_to_params(best_solution)

        print("\n========== BEST RESULT ==========")
        print("Best fitness:", best_fitness)
        print("Best solution:", best_solution)
        print("Best params:", best_params)

        print("\n========== FITNESS HISTORY ==========")
        for gen, best, mean, worst in zip(
            self.fitness_history["generation"],
            self.fitness_history["best_fitness"],
            self.fitness_history["mean_fitness"],
            self.fitness_history["worst_fitness"],
        ):
            print(
                f"Gen {gen}: "
                f"best={best:.6f}, "
                f"mean={mean:.6f}, "
                f"worst={worst:.6f}"
            )

        final_idx = self.modal_pipeline.run(
            best_params,
            room_name="ga_modal_best",
            order=2,
            visualize=False,
            export=False,
        )

        print("Final idx:", final_idx)