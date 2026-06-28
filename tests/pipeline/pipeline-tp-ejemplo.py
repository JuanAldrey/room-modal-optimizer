from room_modal_optimizer.pipeline.pipeline import Pipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator

import numpy as np

room_name = "testing_pipeline"

params = {
    "data": {
        "vertices": {
            "V1": [0.00, 0.00],
            "V2": [6.46, 0.00],
            "V3": [6.46, 4.76],
            "V4": [0.00, 4.76],
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0,
        },
        "audience_area": {
            "V1": [3.00, 1.20],
            "V2": [5.80, 1.20],
            "V3": [5.80, 3.80],
            "V4": [3.00, 3.80],
        },
        "Z": 3.40,
        "source_pos": [[1.00, 2.38, 1.20]],
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

minMicDistance = 0.7

bestMsfd, bestMicPositions = pipeline.run(
    params=params,
    room_name=room_name,
    minMicDistance = minMicDistance,
    nMics=4
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