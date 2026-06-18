from room_modal_optimizer.optimization.optimizer import Optimizer
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator
from room_modal_optimizer.pipeline.pipeline import Pipeline

import numpy as np

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
        "source_pos": [2.5, 3.2, 1.5],
    }
}

# Run GA to find optimized room
optimizer = Optimizer(base_params=base_params, gene_space_config=gene_space_config, minMicDistance=0.25)
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
    params=params,
    room_name="initial_ga",
    minMicDistance = 0.25
)

# Calculate best MSFD for initial room with order 2
meshPathInitial = mesher.create(params, lc=0.28, source_pos=params["data"]["source_pos"], room_name="initial_ga")

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