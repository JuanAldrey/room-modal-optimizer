# Room Modal Optimizer

Room Modal Optimizer is a Python package for acoustic room analysis and optimization.
It combines geometry generation, finite element simulations and genetic algorithms to evaluate and improve low-frequency room behavior.

The project focuses on two main optimization problems:

1. **Room geometry optimization**, where the shape of the room is modified to improve modal response.
2. **Absorber placement optimization**, where resonator or absorber treatments are assigned to surface patches on a fixed room geometry.

The acoustic objective is evaluated using frequency-domain FEM simulations and the MSFD metric.

---

## Features

* 3D room geometry generation from 2D floor vertices, wall inclination angles and room height.
* Gmsh-based mesh generation.
* Direct frequency-domain FEM simulation using DOLFINx.
* MSFD-based acoustic evaluation.
* Genetic algorithm optimization using PyGAD.
* Symmetric room optimization support.
* Patched ceiling and wall geometries for absorber optimization.
* Patch-based impedance assignment for different resonator types.
* JSON output files for each evaluated solution.
* Fitness history, best result summaries and convergence plots.

---

## Project Structure

```text
room_modal_optimizer/
│
├── evaluation/
│   └── evaluator.py
│
├── gui/
│   ├── main.py
│   ├── styles.py
│   └── dummy_functions.py
│
├── meshing/
│   ├── mesher.py
│   └── intersection_validator.py
│
├── optimization/
│   ├── optimizer.py
│   ├── absorption_optimizer.py
│   └── gene_space_validator.py
│
├── pipeline/
│   ├── pipeline.py
│   └── absorption_pipeline.py
│
└── simulation/
    ├── direct_simulator.py
    ├── modal_simulator.py
    ├── microphone.py
    └── resonator_impedances.py
```

### Main modules

#### `meshing`

Contains the geometry and mesh generation tools.

* `Mesher`: builds the 3D room geometry and generates the Gmsh mesh.
* `IntersectionValidator`: checks polygon self-intersections before geometry construction.

#### `simulation`

Contains the FEM simulation tools.

* `DirectSimulator`: solves the frequency-domain Helmholtz problem and computes SPL responses at microphone positions.
* `ModalSimulator`: computes room modes from the FEM eigenvalue problem.
* `Microphone`: handles microphone point evaluation in the FEM domain.
* `resonator_impedances.py`: stores impedance models used for absorber simulations.

#### `evaluation`

Contains acoustic evaluation metrics.

* `Evaluator`: computes response deviations and the MSFD metric.

#### `pipeline`

Contains high-level evaluation workflows.

* `Pipeline`: evaluates a room geometry by generating a mesh, simulating the response and selecting the best microphone configuration.
* `AbsorptionPipeline`: evaluates an absorber configuration on an already patched room mesh.

#### `optimization`

Contains genetic optimization routines.

* `Optimizer`: optimizes room geometry using a genetic algorithm.
* `AbsorptionOptimizer`: optimizes absorber placement and resonator assignment on patched surfaces.
* `GeneSpaceValidator`: checks whether a geometry gene space can generate invalid rooms.

#### `gui`

Contains a graphical interface that can be ran using:

```bash
python -m src.room_modal_optimizer.gui.main
```

#### `tests`

Contains test scripts and validation utilities used during development.

---

## Installation

This project is intended to run inside **WSL/Ubuntu** using a Conda environment.

The recommended setup is:

* Windows + WSL/Ubuntu
* VS Code with the WSL extension
* Conda environment named `room-opt-complex`
* DOLFINx/FEniCSx installed from `conda-forge`
* Project installed in editable mode with `pip install -e .`

---

### 1. Install WSL and Ubuntu

Open PowerShell as administrator and run:

```bash
wsl --install -d Ubuntu
```

Restart the computer if Windows asks for it.

Then open Ubuntu from the Start Menu and create the Linux username and password.

---

### 2. Update Ubuntu packages

Inside the Ubuntu terminal, run:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y git wget curl build-essential
```

---

### 3. Install Conda

If Conda is not installed yet, install Miniforge inside WSL:

```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Accept the installer options.

Then close and reopen the Ubuntu terminal.

Check that Conda works:

```bash
conda --version
```

---

### 4. Create the project environment

Create the Conda environment:

```bash
conda create -n room-opt-complex -c conda-forge python=3.12
```

Activate it:

```bash
conda activate room-opt-complex
```

---

### 5. Install FEM dependencies

Install DOLFINx/FEniCSx with complex PETSc support:

```bash
conda install -c conda-forge fenics-dolfinx mpich "petsc=*=complex*"
```

Install additional project dependencies:

```bash
conda install -c conda-forge gmsh numpy scipy matplotlib pandas
python -m pip install pygad
```

---

### 6. Verify DOLFINx installation

Run:

```bash
python -c "import dolfinx; print('dolfinx OK')"
```

Check that PETSc is using complex numbers:

```bash
python -c "from petsc4py import PETSc; import numpy as np; print(PETSc.ScalarType)"
```

The output should be similar to:

```text
<class 'numpy.complex128'>
```

If PETSc is not complex, the direct Helmholtz simulations may fail or give incorrect behavior.

---

### 7. Clone the repository

Move to the WSL home folder:

```bash
cd ~
```

Clone the repository:

```bash
git clone https://github.com/JuanAldrey/room-modal-optimizer
cd room-modal-optimizer
```

---

### 8. Install the package in editable mode

Make sure the environment is active:

```bash
conda activate room-opt-complex
```

From the project root folder, run:

```bash
python -m pip install -e .
```

Verify that the package imports correctly:

```bash
python -c "import room_modal_optimizer; print('OK')"
```

If this prints `OK`, the package is installed correctly.

---

### 9. Open the project in VS Code

Always open VS Code from the WSL terminal, inside the project folder:

```bash
cd ~/room-modal-optimizer
conda activate room-opt-complex
code .
```

VS Code should open connected to WSL. The bottom-left corner should indicate something like:

```text
WSL: Ubuntu
```

Do not open the project from the Windows file explorer or from a path like:

```text
C:\Users\...
```

The project should be opened from the Linux path:

```text
~/room-modal-optimizer
```

---

### 10. Select the correct Python interpreter in VS Code

Inside VS Code:

1. Press `Ctrl + Shift + P`
2. Search for `Python: Select Interpreter`
3. Select the interpreter from the Conda environment:

```text
room-opt-complex
```

If VS Code does not detect the environment automatically, close VS Code and reopen it from WSL using:

```bash
cd ~/room-modal-optimizer
conda activate room-opt-complex
code .
```

---

### 11. Test the project from VS Code

Open a VS Code terminal and check that the environment is active:

```bash
conda activate room-opt-complex
```

Then run:

```bash
python -c "import room_modal_optimizer; print('OK')"
python -c "import dolfinx; print('dolfinx OK')"
python -c "import gmsh; print('gmsh OK')"
python -c "import pygad; print('pygad OK')"
```

If all commands print `OK`, the installation is ready.

---

## Common Issues

### VS Code does not recognize `room_modal_optimizer`

Make sure the project was installed in editable mode:

```bash
cd ~/room-modal-optimizer
conda activate room-opt-complex
python -m pip install -e .
```

Then select the correct interpreter in VS Code:

```text
Python: Select Interpreter → room-opt-complex
```

---

### The terminal imports the package, but VS Code shows import errors

This usually means VS Code is using the wrong Python interpreter.

Fix it by opening VS Code from WSL:

```bash
cd ~/room-modal-optimizer
conda activate room-opt-complex
code .
```

Then select the `room-opt-complex` interpreter.

---

### DOLFINx works, but direct simulations fail

Check that PETSc was installed with complex scalar support:

```bash
python -c "from petsc4py import PETSc; print(PETSc.ScalarType)"
```

The output should be:

```text
<class 'numpy.complex128'>
```

If it is not complex, recreate the environment and reinstall DOLFINx with:

```bash
conda install -c conda-forge fenics-dolfinx mpich "petsc=*=complex*"
```

---

### MPI issues

Check that `mpirun` comes from the Conda environment:

```bash
which mpirun
```

It should point to a path inside the Conda environment, not to the system installation.

For example:

```text
.../envs/room-opt-complex/bin/mpirun
```

---

## Basic Room Input Format

Room geometries are defined using a dictionary containing floor vertices, wall angles, room height, source positions and audience area.

Example:

```python
base_params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [5.0, 0.0],
            "V3": [5.0, 4.0],
            "V4": [0.0, 4.0],
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0,
        },
        "audience_area": {
            "V1": [1.6, 1.1],
            "V2": [1.6, 2.3],
            "V3": [3.4, 2.3],
            "V4": [3.4, 1.1],
        },
        "Z": 3.0,
        "source_pos": [[2.5, 3.2, 1.5], [2.7, 3.2, 1.5]],
    }
}
```

Wall angles are defined in degrees. The room height is defined by `Z`.

Gene spaces are defined following the base params structure, indicating a range of values the GA can randomly select from.

Example:

```python
gene_space_config = {
    "vertices": {
        "V1": {"dx": [-0.20, 0.20], "dy": [-0.20, 0.20]},
        "V2": {"dx": [-0.20, 0.20], "dy": [-0.20, 0.20]},
        "V3": {"dx": [-0.20, 0.20], "dy": [-0.20, 0.20]},
        "V4": {"dx": [-0.20, 0.20], "dy": [-0.20, 0.20]},
    },
    "walls": {},
    "Z": {"low": 3.0, "high": 4.2}
}
```

For symmetric rooms, ony vertices from one side should be specified, and angles for walls crossing the axis are not allowed.

---

## Geometry Optimization Example

```python
from room_modal_optimizer.optimization.optimizer import Optimizer

optimizer = Optimizer(
    base_params=base_params,
    gene_space_config=gene_space_config,
    minMicDistance=0.5,
    keepSymmetry=True,
    sol_per_pop=10,
    n_generations=100,
    random_seed=42,
)

bestResults = optimizer.run()

bestResult = bestResults[0]
params = bestResult["params"]
bestMicPositions = bestResult["best_mic_positions"]
```

The geometry optimizer encodes room vertices, wall angles and height into a genetic algorithm chromosome. Each solution is converted into a room geometry, simulated with the direct FEM solver and evaluated using MSFD.

---

## Absorber Optimization Example

```python
from room_modal_optimizer.optimization.absorption_optimizer import AbsorptionOptimizer

absorptionOptimizer = AbsorptionOptimizer(
    params=params,
    mic_positions=bestMicPositions,
    percentage=30,
    sol_per_pop=10,
    n_generations=100,
    random_seed=42,
)

bestAbsorptionResults = absorptionOptimizer.run()

bestAbsorptionResult = bestAbsorptionResults[0]
impedanceMappings = bestAbsorptionResult["impedance_mappings"]
```

The absorber optimizer works on a fixed room geometry. It generates a patched mesh, selects a percentage of available patches, assigns resonator types to those patches and evaluates the resulting acoustic response.

Unselected patches are assigned the default impedance condition.

---

## Genetic Algorithm Configuration

Both optimization workflows use PyGAD.

The default GA configuration includes:

* Tournament parent selection.
* Two-point crossover.
* Random mutation.
* Elitism.
* Fixed random seed support.
* Saturation-based stopping criterion.
* Fitness history tracking.
* Best solution saving.

The fitness value is computed from the acoustic objective as:

```python
fitness = 1 / (1 + abs(idx))
```

where `idx` is the objective value returned by the evaluation pipeline. Lower objective values therefore produce higher fitness values.

---

## Output Files

Each GA run creates a result directory under:

```text
ga_results/
```

Typical output files include:

```text
fitness_history.json
best_result.json
run_summary.json
plots/
solutions/
```

### `fitness_history.json`

Stores generation-level statistics:

* best fitness
* mean fitness
* worst fitness

### `best_result.json`

Stores the best solution found during the run.

### `run_summary.json`

Stores general run metadata, including:

* run name
* total runtime
* number of generations
* population size
* number of genes
* best objective value
* best fitness value

### `solutions/`

Contains one JSON file per evaluated solution.

### `plots/`

Contains GA convergence plots, such as:

* `fitness.png`
* `genes_best.png`

---

## Acoustic Evaluation

The main evaluation metric is MSFD, which combines:

* Magnitude Deviation
* Spatial Deviation

The metric is computed from SPL responses at multiple microphone positions across the simulated frequency range.

---

## Notes and Limitations

* The mesh generation process assumes valid, non-self-intersecting room polygons.
* The gene space validator checks extreme combinations before optimization, but invalid geometries may still require additional handling depending on the search space.
* Absorber optimization assumes a fixed patched geometry.
* Patch size is currently based on approximately 1 m² surface regions.
* Source boundaries are modeled using small spherical source surfaces.
* Direct FEM simulations can be computationally expensive, especially for large meshes, high polynomial order or many frequency samples.

---

## References

This project is based on tools and methods commonly used in computational room acoustics, including:

* Gmsh for mesh generation.
* DOLFINx/FEniCSx for finite element simulation.
* PyGAD for genetic optimization.
* Frequency-domain FEM methods for room acoustic analysis.
* MSFD-based response evaluation.
