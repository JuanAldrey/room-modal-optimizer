from room_modal_optimizer.pipeline.pipeline import Pipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator

room_name = 'testing_pipeline'
params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [0.0, 5.0],
            "V3": [3.0, 5.0],
            "V4": [3.0, 0.0]

        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0
        },
        "audience_area": {
            "V1": [1.0, 0.0],
            "V2": [1.0, 2.0],
            "V3": [2.0, 2.0],
            "V4": [2.0, 0.0]
        },
        "Z": 3.0,
        "source_pos": [1.5, 4, 1.5]
    }
}
mesher = Mesher()
modalSimulator = ModalSimulator()
directSimulator = DirectSimulator()
evaluator = Evaluator()

pipeline = Pipeline(mesher, modalSimulator, directSimulator, evaluator)
msfd = pipeline.run(params, room_name=room_name)

print("MSFD: ", msfd)