import copy
import numpy as np
import pygad
import os
import time
import json
from pathlib import Path

from room_modal_optimizer.pipeline.pipeline import Pipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator
from room_modal_optimizer.optimization.gene_space_validator import GeneSpaceValidator

class Optimizer:
    def __init__(
        self,
        base_params,
        gene_space_config,
        sol_per_pop=12,
        n_generations=50,
        minMicDistance=0.25,
        keepSymmetry=False,
        runName=None,
        savePlots=True,
        random_seed=42
    ):
        self.total_runtime_s = None
        self.keepSymmetry = keepSymmetry
        self.base_params = base_params
        self.ga_instance = None
        self.genes = []
        self.gene_space_validator = GeneSpaceValidator()
        self.gene_space = self.define_gene_space(gene_space_config)
        self.sol_per_pop = sol_per_pop
        self.n_generations = n_generations
        self.minMicDistance = minMicDistance
        self.savePlots = savePlots
        self.random_seed=random_seed

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

        if self.keepSymmetry:
            params = self.expandSymmetricParams(params)

        return params
    
    def expandSymmetricParams(self, params, axisTolerance=1e-9):
        params = copy.deepcopy(params)

        vertices = params["data"]["vertices"]
        baseVertices = self.base_params["data"]["vertices"]

        for gene in self.genes:
            if gene["type"] != "vertex":
                continue

            masterKey = gene["key"]

            mirrorKey = self.findMirrorVertexKey(
                masterKey=masterKey,
                baseVertices=baseVertices,
                axisTolerance=axisTolerance,
            )

            masterPoint = np.asarray(vertices[masterKey], dtype=float)

            vertices[mirrorKey] = [
                float(-masterPoint[0]),
                float(masterPoint[1]),
            ]

        params["data"]["vertices"] = vertices

        walls = params["data"].get("walls", {})

        for gene in self.genes:
            if gene["type"] != "wall":
                continue

            masterWallKey = gene["key"]

            mirrorWallKey = self.findMirrorWallKey(
                masterWallKey=masterWallKey,
                baseVertices=baseVertices,
                axisTolerance=axisTolerance,
            )

            walls[mirrorWallKey] = float(walls[masterWallKey])

        params["data"]["walls"] = walls

        return params
    
    def findMirrorVertexKey(self, masterKey, baseVertices, axisTolerance=1e-9):
        masterPoint = np.asarray(baseVertices[masterKey], dtype=float)
        targetPoint = np.asarray([-masterPoint[0], masterPoint[1]], dtype=float)

        for vertexKey, point in baseVertices.items():
            if vertexKey == masterKey:
                continue

            point = np.asarray(point, dtype=float)

            if np.linalg.norm(point - targetPoint) <= axisTolerance:
                return vertexKey

        raise ValueError(
            f"No se encontró vértice espejo para {masterKey}. "
            f"Esperaba un punto cercano a {targetPoint.tolist()}"
        )
    
    def findMirrorWallKey(self, masterWallKey, baseVertices, axisTolerance=1e-9):
        wallEdges = self.getWallEdges(baseVertices)

        p1, p2 = wallEdges[masterWallKey]

        targetP1 = np.asarray([-p1[0], p1[1]], dtype=float)
        targetP2 = np.asarray([-p2[0], p2[1]], dtype=float)

        for wallKey, (q1, q2) in wallEdges.items():
            sameDirection = (
                np.linalg.norm(q1 - targetP1) <= axisTolerance
                and np.linalg.norm(q2 - targetP2) <= axisTolerance
            )

            oppositeDirection = (
                np.linalg.norm(q1 - targetP2) <= axisTolerance
                and np.linalg.norm(q2 - targetP1) <= axisTolerance
            )

            if sameDirection or oppositeDirection:
                return wallKey

        raise ValueError(
            f"No se encontró pared espejo para {masterWallKey}."
        )


    def getWallEdges(self, vertices):
        keys = sorted(vertices.keys(), key=lambda key: int(key[1:]))
        edges = {}

        for i, key in enumerate(keys):
            nextKey = keys[(i + 1) % len(keys)]

            wallKey = f"W{i + 1}"

            p1 = np.asarray(vertices[key], dtype=float)
            p2 = np.asarray(vertices[nextKey], dtype=float)

            edges[wallKey] = (p1, p2)

        return edges
    
    def define_gene_space(self, gene_space_config):
        self.validate_gene_space(gene_space_config)

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
    
    def validate_gene_space(self, gene_space_config):
        ok, message = self.gene_space_validator.validateGeneSpaceSafety(
            baseParams=self.base_params,
            geneSpaceConfig=gene_space_config,
            margin=0.1,
            keepSymmetry=self.keepSymmetry,
        )

        print("Gene space safety: ", ok)
        print(message)

        if not ok:
            raise ValueError(message)
    
    def fitness_func(self, ga_instance, solution, solution_idx):
        generation = ga_instance.generations_completed
        room_name = (
            f"ga_modal_gen_{generation:03d}"
            f"_sol_{solution_idx:03d}"
            f"_pid_{os.getpid()}"
        )

        params = self.solution_to_params(solution)

        pipeline = self.build_pipeline()

        idx, bestMicPositions = pipeline.run(
            params,
            room_name=room_name,
            minMicDistance=self.minMicDistance
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
                "best_mic_positions": None
                if bestMicPositions is None
                else np.asarray(bestMicPositions, dtype=float).tolist(),
                "params": params
            }, f, indent=4)

        return fitness
    
    def build_pipeline(self):
        return Pipeline(
            Mesher(),
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
        print("Microphone positions: ")
        print(bestResult["best_mic_positions"])

        return bestResult["params"], bestResult["best_mic_positions"]
    
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
            "minMicDistance": self.minMicDistance,
            "keepSymmetry": self.keepSymmetry,
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
