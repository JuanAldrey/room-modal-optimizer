from room_modal_optimizer.optimization.optimizer import Optimizer
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator
from room_modal_optimizer.pipeline.pipeline import Pipeline

import numpy as np

gene_space_config = {
    "vertices": {
        "V2": {"dx": [-0.50, 0.50], "dy": [-0.50, 0.50]},
        "V3": {"dx": [-0.50, 0.50], "dy": [-0.50, 0.50]},
    },
    "walls": {
        "W2": {"low": -5.0, "high": 5.0},
    },
    "Z": {"low": 3.0, "high": 4.2},
}


base_params = {
    "data": {
        # Sala completa, orden anticlockwise
        "vertices": {
            "V1": [-2.5, 0.0],
            "V2": [ 2.5, 0.0],
            "V3": [ 2.5, 4.0],
            "V4": [-2.5, 4.0],
        },

        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0,
        },

        # Audience area centrada y bien adentro
        "audience_area": {
            "V1": [-1.0, 1.0],
            "V2": [ 1.0, 1.0],
            "V3": [ 1.0, 2.5],
            "V4": [-1.0, 2.5],
        },

        "Z": 3.0,

        # Fuente centrada en x, no pegada a pared
        "source_pos": [0.0, 3.3, 1.5],
    }
}

# Run GA to find optimized room
optimizer = Optimizer(base_params=base_params, gene_space_config=gene_space_config, minMicDistance=0.25, keepSymmetry=True)
params, micPositions = optimizer.run()

# Calculate best mic positions for initial room with pipeline
mesher = Mesher()
directSimulator = DirectSimulator()
evaluator = Evaluator()
pipeline = Pipeline(
    mesher=mesher,
    directSimulator=directSimulator,
    evaluator=evaluator,
)

msfdInitial, initialBestMicPositions = pipeline.run(
    params=base_params,
    room_name="initial_ga",
    minMicDistance = 0.25
)

# Calculate best MSFD for initial room with order 2
meshPathInitial = mesher.create(base_params, lc=0.28, source_pos=params["data"]["source_pos"], room_name="initial_ga")

freqsOut, initialSplResponses = directSimulator.simulate(
    mesh_path=meshPathInitial,
    mic_positions=initialBestMicPositions,
    order=2,
    room_name="initial_ga",
    freqs=np.arange(20.0, 201.0, 2.0),
    use_impedance=True,
    wall_z=25.0 + 0j,
    floor_z=25.0 + 0j,
    ceiling_z=25.0 + 0j,
)

msfdInitial = evaluator.evaluate_msfd(
        response=initialSplResponses,
        input_is_db=True,
        weight_magnitude=0.5,
        weight_spatial=0.5,
    )["MSFD"]

print("Initial MSFD (order 2): ", msfdInitial)

# Calculate best MSFD for final room with order 2
meshPathFinal = mesher.create(params, lc=0.28, source_pos=params["data"]["source_pos"], room_name="final_ga")

freqsOut, finalSplResponses = directSimulator.simulate(
    mesh_path=meshPathFinal,
    mic_positions=micPositions,
    order=2,
    room_name="final_ga",
    freqs=np.arange(20.0, 201.0, 2.0),
    use_impedance=True,
    wall_z=25.0 + 0j,
    floor_z=25.0 + 0j,
    ceiling_z=25.0 + 0j,
)

msfdFinal = evaluator.evaluate_msfd(
        response=finalSplResponses,
        input_is_db=True,
        weight_magnitude=0.5,
        weight_spatial=0.5,
    )["MSFD"]

print("Best MSFD (order 2): ", msfdFinal)