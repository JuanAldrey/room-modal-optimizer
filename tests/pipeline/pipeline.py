from room_modal_optimizer.pipeline.pipeline import Pipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator

import numpy as np

room_name = "testing_pipeline"

params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [0.0, 5.0],
            "V3": [3.0, 5.0],
            "V4": [3.0, 0.0],
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0,
        },
        "audience_area": {
            "V1": [1.0, 0.0],
            "V2": [1.0, 2.0],
            "V3": [2.0, 2.0],
            "V4": [2.0, 0.0],
        },
        "Z": 3.0,
        "source_pos": [1.5, 4.0, 1.5],
    }
}

mesher = Mesher()
directSimulator = DirectSimulator()
evaluator = Evaluator()

pipeline = Pipeline(
    mesher=mesher,
    directSimulator=directSimulator,
    evaluator=evaluator,
)

minMicDistance = 0.25

bestMsfd, bestMicPositions = pipeline.run(
    params=params,
    room_name=room_name,
    minMicDistance = minMicDistance
)

print("Best MSFD:", bestMsfd)
print("Best mic positions:")
print(bestMicPositions)

order_2 = True

if order_2:

    meshPath = mesher.create(params, lc=0.28, source_pos=params["data"]["source_pos"], room_name=room_name)

    freqsOut, splResponses = directSimulator.simulate(
        mesh_path=meshPath,
        mic_positions=bestMicPositions,
        order=2,
        room_name=room_name,
        freqs=np.arange(20.0, 201.0, 2.0),
        use_impedance=True,
        wall_z=25.0 + 0j,
        floor_z=25.0 + 0j,
        ceiling_z=25.0 + 0j,
    )

    msfd = evaluator.evaluate_msfd(
            response=splResponses,
            input_is_db=True,
            weight_magnitude=0.5,
            weight_spatial=0.5,
        )["MSFD"]
    
    print("Best MSFD (order 2): ", msfd)