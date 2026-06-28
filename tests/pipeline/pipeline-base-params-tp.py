from room_modal_optimizer.pipeline.pipeline import Pipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator

import numpy as np


room_name = "verify_bruteforce_best_pipeline"

params = {
    "data": {
        "vertices": {
            "V1": [-2.75, 0.25],
            "V2": [ 2.75, 0.25],
            "V3": [ 2.25, 3.75],
            "V4": [-2.25, 3.75],
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0,
        },
        "audience_area": {
            "V1": [-1.0, 1.0],
            "V2": [ 1.0, 1.0],
            "V3": [ 1.0, 2.5],
            "V4": [-1.0, 2.5],
        },
        "Z": 3.6,
        "source_pos": [[0.0, 3.0, 1.5]],
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

minMicDistance = 0.5

bestMsfd, bestMicPositions = pipeline.run(
    params=params,
    room_name=room_name,
    minMicDistance=minMicDistance,
    nMics=4,
)

print("Best MSFD brute force params (order 1):", bestMsfd)
print("Best mic positions:")
print(bestMicPositions)

print()
print("Expected brute force MSFD:", 2.6249825991992437)
print("Expected brute force mic positions:")
print(np.array([
    [ 0.75, 1.05, 1.2],
    [-0.85, 1.15, 1.2],
    [-0.25, 1.15, 1.2],
    [ 0.25, 1.15, 1.2],
]))


order_2 = True

if order_2:

    room_name_order_2 = "verify_bruteforce_best_order2"

    meshPath = mesher.create(
        params,
        lc=0.28,
        source_pos=params["data"]["source_pos"],
        room_name=room_name_order_2,
    )

    freqsOut, splResponses = directSimulator.simulate(
        mesh_path=meshPath,
        mic_positions=bestMicPositions,
        order=2,
        room_name=room_name_order_2,
        freqs=np.arange(20.0, 201.0, 2.0),
        use_impedance=True,
        wall_z=25.0 + 0j,
        floor_z=25.0 + 0j,
        ceiling_z=25.0 + 0j,
    )

    msfdOrder2 = evaluator.evaluate_msfd(
        response=splResponses,
        input_is_db=True,
        weight_magnitude=0.5,
        weight_spatial=0.5,
    )["MSFD"]

    print("Best MSFD brute force params (order 2):", msfdOrder2)