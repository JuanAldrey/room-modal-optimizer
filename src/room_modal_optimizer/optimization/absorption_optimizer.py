import numpy as np
import pygad
import os
import time
import json
import math
from pathlib import Path

from room_modal_optimizer.pipeline.absorption_pipeline import AbsorptionPipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator

class AbsorptionOptimizer:
    def __init__(
        self,
        params,
        percentage,
        mic_positions,
        sol_per_pop=12,
        n_generations=50,
        runName=None,
        savePlots=True,
        random_seed=42
    ):
        self.total_runtime_s = None
        self.ga_instance = None
        self.sol_per_pop = sol_per_pop
        self.n_generations = n_generations
        self.savePlots = savePlots
        self.random_seed=random_seed
        self.gene_space = None
        self.gene_constraint = None

        self.params = params
        self.percentage = percentage
        self.mic_positions = mic_positions
        self.n_panels = 0
        self.mesh_path = None
        self.physical_tags = None
        self.mesher = Mesher()

        self.resultsDir = Path("ga_results")
        self.resultsDir.mkdir(parents=True, exist_ok=True)

        if runName is None:
            runName = time.strftime(f"ga_%Y%m%d_%H%M%S_pid_{os.getpid()}")

        self.runName = runName
        self.runDir = self.resultsDir / self.runName
        self.solutionsDir = self.runDir / "solutions"
        self.plotsDir = self.runDir / "plots"

        self.solutionsDir.mkdir(parents=True, exist_ok=True)
        self.plotsDir.mkdir(parents=True, exist_ok=True)

        self.fitness_history = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "worst_fitness": [],
        }

    def run(self):
        print(f"\nGA results directory: {self.runDir}\n")

        self.generate_mesh()
        self.define_gene_space()
        self.create_ga_instance()

        startTime = time.perf_counter()

        self.ga_instance.run()

        endTime = time.perf_counter()
        self.total_runtime_s = endTime - startTime

        print("\n========== GA RUNTIME ==========")
        print(f"Total runtime: {self.total_runtime_s:.2f} s")
        print(f"Total runtime: {self.total_runtime_s / 60:.2f} min")
        print("================================\n")

        bestResults = self.get_best_results()
        bestResult = bestResults[0]

        self.save_run_outputs(bestResult)

        return bestResults

    def generate_mesh(self):
        self.mesh_path, self.physical_tags, patchedArea = self.mesher.create(
            self.params,
            room_name=self.runName,
            visualize=False,
            source_pos=self.params["data"]["source_pos"],
            patch=True
        )

        self.physical_tags = [
            int(tag)
            for name, tag in sorted(
                self.physical_tags.items(),
                key=lambda item: item[1]
            )
        ]

        percentage = self.percentage
        if percentage > 1.0:
            percentage = percentage / 100.0

        targetAbsorberArea = percentage * patchedArea

        self.n_panels = math.floor(targetAbsorberArea / 1.0)
        self.n_panels = min(self.n_panels, len(self.physical_tags))

        if self.n_panels <= 0:
            raise ValueError("n_panels is 0. Increase percentage or check patch generation.")


    def solution_to_impedance_mappings(self, solution):
        solution = np.asarray(solution, dtype=int)

        patchGenes = solution[:self.n_panels]
        resonatorGenes = solution[self.n_panels:2 * self.n_panels]

        impedanceMappings = {
            int(physicalTag): 0
            for physicalTag in self.physical_tags
        }

        for patchIndex, resonatorType in zip(patchGenes, resonatorGenes):
            physicalTag = self.physical_tags[int(patchIndex)]
            impedanceMappings[int(physicalTag)] = int(resonatorType)

        return impedanceMappings
    
    def define_gene_space(self):
        numPatches = len(self.physical_tags)

        patchGeneSpace = list(range(numPatches))
        resonatorGeneSpace = [1, 2, 3]

        self.gene_space = (
            [patchGeneSpace.copy() for _ in range(self.n_panels)]
            + [resonatorGeneSpace.copy() for _ in range(self.n_panels)]
        )
    
    def make_patch_constraint(self, geneIndex):
        def patch_constraint(solution, values):
            selectedPatches = []

            for i in range(self.n_panels):
                if i == geneIndex:
                    continue

                selectedPatches.append(int(solution[i]))

            allowedValues = [
                value
                for value in values
                if int(value) not in selectedPatches
            ]

            return allowedValues

        return patch_constraint
    
    def create_gene_constraint(self):
        patchConstraints = [
            self.make_patch_constraint(i)
            for i in range(self.n_panels)
        ]

        resonatorConstraints = [
            None
            for _ in range(self.n_panels)
        ]

        return patchConstraints + resonatorConstraints

    def fitness_func(self, ga_instance, solution, solution_idx):
        generation = ga_instance.generations_completed
        room_name = (
            f"ga_modal_gen_{generation:03d}"
            f"_sol_{solution_idx:03d}"
            f"_pid_{os.getpid()}"
        )

        impedance_mappings = self.solution_to_impedance_mappings(solution)

        pipeline = self.build_pipeline()

        idx = pipeline.run(
            mesh_path=self.mesh_path,
            impedance_mappings=impedance_mappings,
            mic_positions=self.mic_positions,
            room_name=room_name,
        )
        
        if idx is None:
            fitness = 1e-9
            print(f"{room_name} | idx=None | fitness={fitness:.6f}")
        else:
            fitness = 1.0 / (1.0 + abs(idx))
            print(f"{room_name} | idx={idx:.6f} | fitness={fitness:.6f}")

        with open(self.solutionsDir / f"{room_name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "room_name": room_name,
                "generation": int(generation),
                "solution_idx": int(solution_idx),
                "idx": None if idx is None else float(idx),
                "fitness": float(fitness),
                "impedance_mappings": impedance_mappings
            }, f, indent=4)

        return fitness
    
    def build_pipeline(self):
        return AbsorptionPipeline (
            DirectSimulator(),
            Evaluator()
        )
    
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
    
    def create_ga_instance(self):
        if self.sol_per_pop < 3:
            raise ValueError("sol_per_pop must be at least 3 for GA optimization.")

        self.ga_instance = pygad.GA(
            num_generations=self.n_generations,
            sol_per_pop=self.sol_per_pop,
            num_parents_mating = min(6, self.sol_per_pop - 1),

            num_genes=len(self.gene_space),
            gene_space=self.gene_space,
            gene_constraint=self.create_gene_constraint(),

            fitness_func=self.fitness_func,

            parent_selection_type="tournament",
            K_tournament=3,

            crossover_type="two_points",
            crossover_probability=0.85,

            mutation_type="random",
            mutation_probability=0.15,

            keep_elitism=1,

            on_generation=self.on_generation,

            random_seed=self.random_seed,
            stop_criteria=["saturate_12"],
            save_best_solutions=True,
        )

    def get_history(self):
        print("\n========== BEST ROOM ==========")
        bestResult = self.get_best_result()

        print("Room name: ", bestResult["room_name"])
        print("MSFD (order 1): ", bestResult["idx"])

        return bestResult["room_name"], bestResult["idx"]
    
    def get_best_results(self):
        history = []

        for resultPath in self.solutionsDir.glob("*.json"):
            with open(resultPath, "r", encoding="utf-8") as f:
                result = json.load(f)

            history.append(result)

        history = sorted(
            history,
            key=lambda item: item["fitness"],
            reverse=True,
        )

        if len(history) == 0:
            raise RuntimeError("No GA results were generated.")

        return history[0:10]
    
    def save_run_outputs(self, bestResult):
        with open(self.runDir / "fitness_history.json", "w", encoding="utf-8") as f:
            json.dump(self.fitness_history, f, indent=4)

        with open(self.runDir / "best_result.json", "w", encoding="utf-8") as f:
            json.dump(bestResult, f, indent=4)

        summary = {
            "run_name": self.runName,
            "total_runtime_s": self.total_runtime_s,
            "total_runtime_min": self.total_runtime_s / 60.0,
            "sol_per_pop": self.sol_per_pop,
            "n_generations": self.n_generations,
            "num_genes": len(self.gene_space),
            "best_room_name": bestResult["room_name"],
            "best_idx": bestResult["idx"],
            "best_fitness": bestResult["fitness"],
            "results_dir": str(self.runDir),
        }

        with open(self.runDir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        if self.savePlots:
            self.save_ga_plots()

    def save_ga_plots(self):
        self.ga_instance.plot_fitness(
            title="GA fitness convergence",
            xlabel="Generation",
            ylabel="Fitness",
            save_dir=str(self.plotsDir / "fitness.png"),
        )

        self.ga_instance.plot_genes(
            solutions="best",
            graph_type="plot",
            title="Best solution genes evolution",
            xlabel="Generation",
            ylabel="Gene value",
            save_dir=str(self.plotsDir / "genes_best.png"),
        )
