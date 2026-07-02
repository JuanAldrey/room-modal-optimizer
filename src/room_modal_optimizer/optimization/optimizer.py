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
    """
    Optimizes room geometry using a genetic algorithm.

    The optimizer encodes room variables such as vertex positions, wall angles
    and room height into a PyGAD chromosome. Each candidate solution is decoded
    into room parameters, evaluated through the geometry simulation pipeline, and
    assigned a fitness value based on the resulting MSFD metric.

    The class supports symmetric room optimization, where only part of the geometry
    is encoded in the chromosome and the mirrored side is reconstructed
    automatically. It also saves each evaluated solution, tracks generation-level
    fitness statistics, stores the best results, and writes summary files and
    plots for each GA run.
    """
    def __init__(
        self,
        base_params,
        gene_space_config,
        sol_per_pop=12,
        n_generations=50,
        minMicDistance=0.25,
        nMics=4,
        fmax=200,
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
        self.nMics = nMics
        self.fmax = fmax
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
        """
        Runs the genetic algorithm optimization process.

        The method creates the PyGAD instance, executes the genetic algorithm, measures
        the total runtime, retrieves the best optimization results, saves the run
        outputs, and returns the best results found.

        Runtime information is stored in self.total_runtime_s and printed in seconds
        and minutes. The main output files are saved in self.runDir.

        Returns:
            list[dict]: Best optimization results returned by get_best_results(),
            ordered from best to worst according to the fitness criterion.
        """
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
        """
        Converts a genetic algorithm solution vector into room parameters.

        The method starts from a deep copy of self.base_params and updates the room
        variables encoded in self.genes. Vertex genes consume two consecutive values
        from the solution vector, corresponding to x and y coordinates. Wall and
        height genes consume one value each.

        If symmetry is enabled, the decoded base parameters are expanded into the full
        symmetric room geometry before returning.

        Args:
            solution (array-like): Genetic algorithm chromosome containing the encoded
                room variables.

        Returns:
            dict: Room parameter dictionary with the decoded geometry values.
        """
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
        """
        Expands a partially encoded room into a full symmetric geometry.

        The method mirrors the vertex and wall genes encoded in the chromosome across
        the x=0 symmetry axis. For each optimized vertex, the corresponding mirrored
        vertex is found using the base geometry, and its coordinates are updated as
        (-x, y). For each optimized wall angle, the corresponding mirrored wall is
        found and assigned the same angle value.

        The input params dictionary is deep-copied before modification, so the original
        dictionary is not mutated.

        Args:
            params (dict): Room parameter dictionary decoded from the GA solution.
            axisTolerance (float): Numerical tolerance used to detect vertices lying
                on the symmetry axis.

        Returns:
            dict: Full room parameter dictionary with mirrored vertices and walls.
        """
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
        """
        Finds the vertex key corresponding to the mirror of a given vertex.

        The mirror search is performed on the base geometry using the x=0 symmetry
        axis. The expected mirrored point is computed as (-x, y), and the method
        returns the key of the vertex whose coordinates match that point within the
        specified tolerance.

        Args:
            masterKey (str): Key of the original vertex to mirror.
            baseVertices (dict): Dictionary of base room vertices.
            axisTolerance (float): Maximum distance allowed when matching the mirrored
                vertex coordinates.

        Returns:
            str: Key of the mirrored vertex.

        Raises:
            ValueError: If no matching mirrored vertex is found.
        """
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
        """
        Finds the wall key corresponding to the mirror of a given wall.

        The method first builds the wall edge dictionary from the base room vertices.
        The selected wall edge is mirrored across the x=0 symmetry axis, transforming
        each endpoint as (x, y) -> (-x, y). The method then searches for a wall whose
        endpoints match the mirrored edge within the specified tolerance.

        The match is accepted whether the mirrored wall has the same endpoint order
        or the opposite endpoint order.

        Args:
            masterWallKey (str): Key of the original wall to mirror.
            baseVertices (dict): Dictionary of base room vertices used to reconstruct
                wall edges.
            axisTolerance (float): Maximum distance allowed when matching mirrored
                wall endpoints.

        Returns:
            str: Key of the mirrored wall.

        Raises:
            ValueError: If no matching mirrored wall is found.
        """
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
        """
        Builds the wall edge dictionary from ordered room vertices.

        Vertices are sorted by their numeric key order, such as V1, V2 and V3. Each
        consecutive vertex pair defines one wall edge, and the last vertex is connected
        back to the first one to close the polygon. Wall keys are generated as W1, W2,
        W3, matching the vertex edge order.

        Args:
            vertices (dict): Dictionary of room vertices, where each key is a vertex
                name and each value is an [x, y] coordinate.

        Returns:
            dict: Dictionary mapping wall keys to edge endpoint pairs. Each value is
            a tuple containing two NumPy arrays: (p1, p2).
        """
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
        """
        Builds the PyGAD gene space from the optimization configuration.

        The method validates the received gene space configuration and converts it
        into the list of lower and upper bounds expected by PyGAD. Vertex genes are
        encoded using two consecutive values, corresponding to x and y coordinate
        offsets from the base geometry. Wall angle genes and room height genes are
        encoded using one value each.

        While building the gene space, the method also fills self.genes with metadata
        describing the meaning of each gene block. This metadata is later used to
        decode a GA solution back into room parameters.

        Args:
            gene_space_config (dict): Optimization bounds for vertices, walls and
                room height. Vertex bounds are defined as dx and dy offsets from
                self.base_params.

        Returns:
            list[dict]: PyGAD-compatible gene space, where each entry defines the
            lower and upper bound of one scalar gene.
        """
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
        """
        Validates the genetic algorithm search space before optimization.

        The method delegates the safety checks to self.gene_space_validator, using the
        current base parameters, the proposed gene space configuration and the current
        symmetry setting. If the configuration can generate unsafe or invalid room
        geometries, the method raises a ValueError and stops the optimization setup.

        Args:
            gene_space_config (dict): Optimization bounds for vertices, walls and
                room height.

        Raises:
            ValueError: If the gene space is considered unsafe or geometrically
            invalid by the validator.

        Returns:
            None
        """
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
        """
        Evaluates the fitness of a genetic algorithm solution.

        The method converts the GA chromosome into room parameters, builds a fresh
        evaluation pipeline, runs the acoustic simulation, and converts the resulting
        objective value into a fitness score. Since the optimizer minimizes the
        acoustic metric indirectly, lower objective values produce higher fitness
        values using:

            fitness = 1 / (1 + abs(idx))

        A JSON file is saved for each evaluated solution, including the generation,
        solution index, objective value, fitness, selected microphone positions and
        decoded room parameters.

        Args:
            ga_instance: Current PyGAD GA instance.
            solution (array-like): Chromosome values for the current candidate room.
            solution_idx (int): Index of the solution within the current generation.

        Returns:
            float: Fitness value used by PyGAD. Invalid or failed evaluations return
            a small positive fitness value.
        """
        generation = ga_instance.generations_completed
        room_name = (
            f"ga_modal_gen_{generation:03d}"
            f"_sol_{solution_idx:03d}"
            f"_pid_{os.getpid()}"
        )

        params = self.solution_to_params(solution)

        pipeline = self.build_pipeline()

        result = pipeline.run(
            params,
            room_name=room_name,
            minMicDistance=self.minMicDistance,
            nMics=self.nMics,
            fmax=200
        )
        
        if result is None:
            fitness = 1e-9
            print(f"{room_name} | idx=None | fitness={fitness:.6f}")
            return fitness
        else:
            idx, bestMicPositions = result
            fitness = 1.0 / (1.0 + abs(idx))
            print(f"{room_name} | idx={idx:.6f} | fitness={fitness:.6f}")

        with open(self.solutionsDir / f"{self.runName}_{room_name}.json", "w", encoding="utf-8") as f:
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
        """
        Creates a fresh geometry evaluation pipeline.

        A new Mesher, DirectSimulator and Evaluator are instantiated for each pipeline
        build. This keeps each GA evaluation isolated and avoids reusing internal
        simulation state between candidate rooms.

        Returns:
            Pipeline: Geometry evaluation pipeline used to compute the objective value
            for one GA solution.
        """
        return Pipeline(
            Mesher(),
            DirectSimulator(),
            Evaluator()
        )
    
    def on_generation(self, ga_instance):
        """
        Records and prints fitness statistics after each GA generation.

        This method is used as a PyGAD generation callback. It reads the fitness
        values from the last completed generation, computes best, mean and worst
        fitness values, stores them in self.fitness_history, and prints a short
        generation summary.

        Args:
            ga_instance: Current PyGAD GA instance.

        Returns:
            None
        """
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
        """
        Creates and configures the PyGAD genetic algorithm instance.

        The method initializes self.ga_instance using the optimizer gene space,
        fitness function and GA hyperparameters. It uses tournament parent selection,
        two-point crossover, random mutation, elitism and a saturation-based stopping
        criterion.

        The generation callback is connected through self.on_generation, and the best
        solutions are saved internally by PyGAD during the optimization.

        Raises:
            ValueError: If sol_per_pop is lower than 3, since the configured GA setup
            requires at least three solutions per population.

        Returns:
            None
        """
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
        """
        Prints and returns the best room configuration found by the optimizer.

        The method retrieves the best saved result, prints its room name, objective
        value and selected microphone positions, and returns the decoded room
        parameters together with the best microphone positions.

        Returns:
            tuple[dict, list | np.ndarray]: Best room parameters and selected
            microphone positions.
        """
        print("\n========== BEST ROOM ==========")
        bestResult = self.get_best_result()

        print("Room name: ", bestResult["room_name"])
        print("MSFD (order 1): ", bestResult["idx"])
        print("Microphone positions: ")
        print(bestResult["best_mic_positions"])

        return bestResult["params"], bestResult["best_mic_positions"]
    
    def get_best_results(self):
        """
        Loads and returns the best saved GA evaluation results.

        The method reads all solution JSON files stored in self.solutionsDir, sorts
        them by fitness in descending order, and returns the top results. This allows
        the optimizer to recover the best candidates from the saved evaluation files
        instead of relying only on the in-memory PyGAD state.

        Returns:
            list[dict]: Top 10 saved GA results sorted from highest to lowest fitness.

        Raises:
            RuntimeError: If no GA result files were found.
        """
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
        """
        Saves the main output files of a completed GA run.

        The method writes the fitness history, the best result and a run summary to
        JSON files inside self.runDir. The summary includes runtime information, GA
        configuration, geometry optimization settings and the best solution metadata.

        If plot saving is enabled, the method also generates the GA convergence plots.

        Args:
            bestResult (dict): Best GA result dictionary, usually obtained from
                get_best_results()[0].

        Returns:
            None
        """
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
        """
        Saves GA convergence and gene evolution plots.

        The method uses PyGAD's built-in plotting utilities to save the fitness
        convergence curve and the evolution of the best solution genes across
        generations. The plots are written inside self.plotsDir.

        Returns:
            None
        """
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
