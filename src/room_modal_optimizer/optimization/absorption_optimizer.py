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
    """
    Optimizes absorber placement and resonator assignment using a genetic algorithm.

    This optimizer works on a fixed room geometry. It first generates a patched
    room mesh, where ceiling and wall surfaces are subdivided into individual
    physical groups. The genetic algorithm then selects which patches receive
    absorber panels and assigns a resonator type to each selected patch.

    Each candidate solution is converted into a mapping between physical patch
    tags and resonator types. This mapping is evaluated through a direct FEM
    simulation, and the resulting SPL responses are scored using the MSFD metric.

    The class prevents duplicate patch selection through gene constraints, tracks
    generation-level fitness statistics, saves each evaluated solution, stores the
    best results, and writes summary files and convergence plots for each GA run.
    """
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
        """
        Runs the absorber genetic optimization process.

        The method generates the patched room mesh, defines the absorber gene space,
        creates the PyGAD instance, runs the genetic algorithm, measures the total
        runtime, retrieves the best absorber configurations, saves the run outputs,
        and returns the best results found.

        Runtime information is stored in self.total_runtime_s and printed in seconds
        and minutes. The main output files are saved in self.runDir.

        Returns:
            list[dict]: Best absorber optimization results returned by
            get_best_results(), ordered from best to worst according to fitness.
        """
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
        """
        Generates the patched room mesh and computes the number of absorber panels.

        The method creates a patched Gmsh mesh using the current room parameters and
        stores the mesh path and available patch physical tags. The physical tags are
        sorted by their numeric value and converted into a list so they can be used as
        candidate patch locations during the absorber optimization.

        The requested absorber percentage is applied to the total patched area returned
        by the mesher. The resulting target area is converted into an integer number
        of 1 m² absorber panels, limited by the number of available patches.

        Raises:
            ValueError: If the computed number of absorber panels is zero.

        Returns:
            None
        """
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
        """
        Converts a GA solution into patch impedance assignments.

        The chromosome is split into two sections. The first section selects the patch
        indices where absorber panels will be placed, and the second section assigns
        a resonator type to each selected patch. All available patches are initialized
        with resonator type 0, representing the default untreated impedance.

        The returned dictionary maps each physical patch tag to its assigned resonator
        type, which is later used by the direct simulator to apply patch-based
        impedance boundary conditions.

        Args:
            solution (array-like): GA chromosome containing patch index genes followed
                by resonator type genes.

        Returns:
            dict[int, int]: Mapping from physical patch tags to resonator type IDs.
        """
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
        """
        Defines the GA gene space for absorber patch selection and resonator assignment.

        The chromosome is composed of two consecutive sections. The first section
        contains self.n_panels genes that select patch indices from the available
        physical tags. The second section contains self.n_panels genes that assign a
        resonator type to each selected patch.

        Patch genes can take any index from the available patch list. Resonator genes
        can take values 1, 2 or 3, corresponding to the available absorber types. The
        untreated/default case is not included here because unselected patches are
        later assigned resonator type 0 in solution_to_impedance_mappings().

        Returns:
            None
        """
        numPatches = len(self.physical_tags)

        patchGeneSpace = list(range(numPatches))
        resonatorGeneSpace = [1, 2, 3]

        self.gene_space = (
            [patchGeneSpace.copy() for _ in range(self.n_panels)]
            + [resonatorGeneSpace.copy() for _ in range(self.n_panels)]
        )
    
    def make_patch_constraint(self, geneIndex):
        """
        Creates a gene constraint that prevents duplicate patch selection.

        The returned constraint function is intended for patch selection genes. It
        removes from the allowed values any patch index that is already selected by
        the other patch genes in the same chromosome. This ensures that each absorber
        panel is assigned to a different physical patch.

        Args:
            geneIndex (int): Index of the patch gene for which the constraint is
                being created.

        Returns:
            callable: Constraint function compatible with PyGAD. The function receives
            the current solution and candidate values, and returns only the values
            that do not duplicate another selected patch.
        """
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
        """
        Creates the gene constraint list for the absorber GA chromosome.

        The first self.n_panels constraints are assigned to the patch selection genes
        and prevent selecting the same patch more than once. The remaining
        self.n_panels entries correspond to resonator type genes and are left
        unconstrained.

        Returns:
            list[callable | None]: Gene constraint list compatible with PyGAD. Patch
            genes receive duplicate-prevention constraints, while resonator genes
            receive None.
        """
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
        """
        Evaluates the fitness of an absorber GA solution.

        The method converts the GA chromosome into patch impedance mappings, builds a
        fresh absorber evaluation pipeline, runs the direct FEM simulation for the
        current absorber configuration, and converts the resulting objective value
        into a fitness score.

        Since the optimizer minimizes the acoustic metric indirectly, lower objective
        values produce higher fitness values using:

            fitness = 1 / (1 + abs(idx))

        If the simulation fails or returns None, a small fallback fitness value is
        assigned. A JSON file is saved for each evaluated solution, including the
        generation, solution index, objective value, fitness and impedance mappings.

        Args:
            ga_instance: Current PyGAD GA instance.
            solution (array-like): Chromosome containing patch selection genes and
                resonator type genes.
            solution_idx (int): Index of the solution within the current generation.

        Returns:
            float: Fitness value used by PyGAD.
        """
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
        """
        Creates a fresh absorber evaluation pipeline.

        A new DirectSimulator and Evaluator are instantiated for each pipeline build.
        This keeps each absorber GA evaluation isolated and avoids reusing internal
        simulation state between candidate impedance configurations.

        Returns:
            AbsorptionPipeline: Pipeline used to evaluate one absorber configuration.
        """
        return AbsorptionPipeline (
            DirectSimulator(),
            Evaluator()
        )
    
    def on_generation(self, ga_instance):
        """
        Records and prints fitness statistics after each absorber GA generation.

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
        Creates and configures the PyGAD instance for absorber optimization.

        The method initializes self.ga_instance using the absorber gene space, gene
        constraints and fitness function. The configured GA uses tournament selection
        with K=3, two-point crossover with probability 0.85, random mutation with
        probability 0.15, one elite solution per generation, a fixed random seed and
        a saturation-based stopping criterion.

        Gene constraints are applied to the patch selection genes to prevent duplicate
        patch assignments, while resonator genes remain unconstrained.

        The generation callback is connected through self.on_generation, and PyGAD is
        configured to store the best solutions found during the optimization.

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
        """
        Prints and returns the best room configuration found by the optimizer.

        The method retrieves the best saved result, prints its room name, objective
        value and selected microphone positions, and returns the decoded room
        parameters together with the best microphone positions.

        Returns:
            tuple[dict, list]: Best room parameters
        """
        print("\n========== BEST ROOM ==========")
        bestResult = self.get_best_result()

        print("Room name: ", bestResult["room_name"])
        print("MSFD (order 1): ", bestResult["idx"])

        return bestResult["room_name"], bestResult["idx"]
    
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
