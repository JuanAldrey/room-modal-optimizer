from room_modal_optimizer.pipeline.modal_pipeline import ModalPipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator
from room_modal_optimizer.evaluation.modal_evaluator import ModalEvaluator

room_name = 'testing_pipeline'
params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [2.0, -0.2],
            "V3": [4.0, 0.3],
            "V4": [4.7, 1.6],
            "V5": [4.0, 3.0],
            "V6": [2.4, 3.5],
            "V7": [0.7, 3.0],
            "V8": [-0.4, 1.4]
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0,
            "W5": 0.0,
            "W6": 0.0,
            "W7": 0.0,
            "W8": 0.0
        },
        "Z": 3.0
    }
}
mesher = Mesher()
modalSimulator = ModalSimulator()
modalEvaluator = ModalEvaluator()

modalPipeline = ModalPipeline(mesher, modalSimulator, modalEvaluator)
idx = modalPipeline.run(params, room_name=room_name)

print("Fitness result: ", idx)