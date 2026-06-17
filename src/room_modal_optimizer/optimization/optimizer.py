import copy
import numpy as np
import pygad
import os
import time

from room_modal_optimizer.pipeline.pipeline import Pipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator

# Gene space config format:
gene_space_config = {
    "vertices": {
        "V1": {"dx": [-0.5, 0.5], "dy": [-0.5, 0.5]},
        "V2": {"dx": [-0.5, 0.5], "dy": [-0.5, 0.5]},
        "V3": {"dx": [-0.5, 0.5], "dy": [-0.5, 0.5]},
        "V4": {"dx": [-0.5, 0.5], "dy": [-0.5, 0.5]},
    },
    "walls": {
        "W1": {"low": -5.0, "high": 5.0},
        "W2": {"low": -5.0, "high": 5.0},
        "W3": {"low": -5.0, "high": 5.0},
        "W4": {"low": -5.0, "high": 5.0},
    },
    "Z": {"low": 2.0, "high": 5.0}
}

base_params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [0.0, 5.0],
            "V3": [3.0, 5.0],
            "V4": [3.0, 0.0]

        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0
        },
        "audience_area": {
            "V1": [1.0, 0.0],
            "V2": [1.0, 2.0],
            "V3": [2.0, 2.0],
            "V4": [2.0, 0.0]
        },
        "Z": 3.0,
        "source_pos": [1.5, 4, 1.5]
    }
}

class Optimizer:
    def __init__(self, base_params, gene_space_config):
        self.total_runtime_s = None
        self.base_params = base_params
        self.ga_instance = None
        self.genes = []
        self.gene_space = self.define_gene_space(gene_space_config)

        self.fitness_history = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "worst_fitness": [],
        }

    def run(self):
        self.create_ga_instance()

        startTime = time.perf_counter()

        self.ga_instance.run()

        endTime = time.perf_counter()
        self.total_runtime_s = endTime - startTime

        print("\n========== GA RUNTIME ==========")
        print(f"Total runtime: {self.total_runtime_s:.2f} s")
        print(f"Total runtime: {self.total_runtime_s / 60:.2f} min")
        print("================================\n")

    def build_pipeline(self):
        return Pipeline(
            Mesher(),
            ModalSimulator(),
            DirectSimulator(),
            Evaluator()
        )

    def solution_to_params(self, solution):
        params = copy.deepcopy(self.base_params)
        solution_idx = 0
        for gene in self.genes:
            if gene["type"] == "vertex":
                params["data"]["vertices"][gene["key"]] = [float(solution[solution_idx]), float(solution[solution_idx + 1])]
                solution_idx += 2
            elif gene["type"] == "wall":
                params["data"]["walls"][gene["key"]] = float(solution[solution_idx])
                solution_idx += 1
            elif gene["type"] == "height":
                params["data"]["Z"] = float(solution[solution_idx])
                solution_idx += 1

        return params
    
    def fitness_func(self, ga_instance, solution, solution_idx):
        generation = ga_instance.generations_completed
        room_name = (
            f"ga_modal_gen_{generation:03d}"
            f"_sol_{solution_idx:03d}"
            f"_pid_{os.getpid()}"
        )

        params = self.solution_to_params(solution)

        pipeline = self.build_pipeline()

        idx = pipeline.run(
            params,
            room_name=room_name,
        )
        
        if idx is None:
            fitness = 1e-9
            print(f"{room_name} | idx=None | fitness={fitness:.6f}")
        else:
            fitness = 1.0 / (1.0 + abs(idx))
            print(f"{room_name} | idx={idx:.6f} | fitness={fitness:.6f}")

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

    def define_gene_space(self, gene_space_config):
        gene_space = []
        for vertex_key, vertex_config in gene_space_config["vertices"].items():
            gene_space.append({
                "low": self.base_params["data"]["vertices"][vertex_key][0] + vertex_config["dx"][0],
                "high": self.base_params["data"]["vertices"][vertex_key][0] + vertex_config["dx"][1]
                })
            gene_space.append({
                "low": self.base_params["data"]["vertices"][vertex_key][1] + vertex_config["dy"][0],
                "high": self.base_params["data"]["vertices"][vertex_key][1] + vertex_config["dy"][1]
                })
            self.genes.append(({"type": "vertex", "key": vertex_key}))
        for wall_key, wall_config in gene_space_config["walls"].items():
            gene_space.append({
                "low": wall_config["low"], 
                "high": wall_config["high"]
                })
            self.genes.append(({"type": "wall", "key": wall_key}))
        if gene_space_config["Z"]:
            gene_space.append({
                "low": gene_space_config["Z"]["low"],
                "high": gene_space_config["Z"]["high"],
            })
            self.genes.append(({"type": "height", "key": "Z"}))

        return gene_space
    
    # TODO: investigate better GA params
    def create_ga_instance(self):
        self.ga_instance = pygad.GA(
            num_generations=2,
            sol_per_pop=3,
            num_parents_mating=2,

            num_genes=len(self.gene_space),
            gene_space=self.gene_space,

            fitness_func=self.fitness_func,

            parallel_processing=["process", 1],

            crossover_type="single_point",

            mutation_type="random",
            mutation_probability=0.15,

            keep_elitism=1,

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

        pipeline = self.build_pipeline()

        final_idx = pipeline.run(
            best_params,
            room_name="ga_modal_best"
        )

        print("Final idx:", final_idx)

if __name__ == "__main__":
    optimizer = Optimizer(base_params, gene_space_config)
    optimizer.run()
    optimizer.get_history()