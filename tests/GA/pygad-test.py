import copy
import traceback
import numpy as np
import pygad

from room_modal_optimizer.pipeline.modal_pipeline import ModalPipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator
from room_modal_optimizer.evaluation.modal_evaluator import ModalEvaluator


BASE_PARAMS = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [2.0, -0.2],
            "V3": [4.0, 0.3],
            "V4": [4.7, 1.6],
            "V5": [4.0, 3.0],
            "V6": [2.4, 3.5],
            "V7": [0.7, 3.0],
            "V8": [-0.4, 1.4],
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
        "Z": 3.0,
    }
}


mesher = Mesher()
modalSimulator = ModalSimulator()
modalEvaluator = ModalEvaluator()

modalPipeline = ModalPipeline(
    mesher,
    modalSimulator,
    modalEvaluator,
)


def solution_to_params(solution):
    params = copy.deepcopy(BASE_PARAMS)

    # Genes 0 a 7: inclinaciones de paredes
    for i in range(8):
        params["data"]["walls"][f"W{i + 1}"] = float(solution[i])

    # Gen 8: altura
    params["data"]["Z"] = float(solution[8])
    
    print(params)

    return params

def fitness_func(ga_instance, solution, solution_idx):
    generation = ga_instance.generations_completed
    room_name = f"ga_modal_gen_{generation:03d}_sol_{solution_idx:03d}"

    params = solution_to_params(solution)

    idx = modalPipeline.run(
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

    print(f"{room_name} | idx={idx:.6f} | fitness={fitness:.6f}")

    return fitness

def on_generation(ga_instance):
    generation = ga_instance.generations_completed
    fitness_values = ga_instance.last_generation_fitness

    best_fitness = np.max(fitness_values)
    mean_fitness = np.mean(fitness_values)
    worst_fitness = np.min(fitness_values)

    fitness_history["generation"].append(generation)
    fitness_history["best_fitness"].append(float(best_fitness))
    fitness_history["mean_fitness"].append(float(mean_fitness))
    fitness_history["worst_fitness"].append(float(worst_fitness))

    print("\n==============================")
    print("Generation:", generation)
    print("Best fitness:", best_fitness)
    print("Mean fitness:", mean_fitness)
    print("Worst fitness:", worst_fitness)
    print("==============================\n")


gene_space = [
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

fitness_history = {
    "generation": [],
    "best_fitness": [],
    "mean_fitness": [],
    "worst_fitness": [],
}

ga_instance = pygad.GA(
    num_generations=100,
    sol_per_pop=10,
    num_parents_mating=5,

    num_genes=len(gene_space),
    gene_space=gene_space,

    fitness_func=fitness_func,

    parent_selection_type="tournament",
    K_tournament=3,

    crossover_type="single_point",

    mutation_type="random",
    mutation_probability=0.15,

    keep_elitism=3,

    on_generation=on_generation,

    random_seed=42,
)

if __name__ == "__main__":
    ga_instance.run()

    best_solution, best_fitness, best_idx = ga_instance.best_solution()
    best_params = solution_to_params(best_solution)

    print("\n========== BEST RESULT ==========")
    print("Best fitness:", best_fitness)
    print("Best solution:", best_solution)
    print("Best params:", best_params)

    print("\n========== FITNESS HISTORY ==========")
    for gen, best, mean, worst in zip(
        fitness_history["generation"],
        fitness_history["best_fitness"],
        fitness_history["mean_fitness"],
        fitness_history["worst_fitness"],
    ):
        print(
            f"Gen {gen}: "
            f"best={best:.6f}, "
            f"mean={mean:.6f}, "
            f"worst={worst:.6f}"
        )

    final_idx = modalPipeline.run(
        best_params,
        room_name="ga_modal_best",
        order=2,
        visualize=False,
        export=False,
    )

    print("Final idx:", final_idx)
    
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(
        fitness_history["generation"],
        fitness_history["best_fitness"],
        marker="o",
        label="Best fitness",
    )

    plt.plot(
        fitness_history["generation"],
        fitness_history["mean_fitness"],
        marker="o",
        label="Mean fitness",
    )

    plt.plot(
        fitness_history["generation"],
        fitness_history["worst_fitness"],
        marker="o",
        label="Worst fitness",
    )

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA fitness history")
    plt.grid(True)
    plt.legend()
    plt.show()
