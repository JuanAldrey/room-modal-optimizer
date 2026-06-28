from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator

room_name = "testing_direct"

params_8_walls_angled = {
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
            "W1": 3.0,
            "W2": 0.0,
            "W3": 0.3,
            "W4": 5.0,
            "W5": 0.0,
            "W6": 5.0,
            "W7": -3.0,
            "W8": -3.0
        },
        "Z": 3.0
    }
}

# lc chosen from highest frequency:
# lambda_min = c / f_max = 343 / 200 = 1.715 m
# Use ~6 elems per wavelength:
# lc = 1.715 / 6 = 0.286 m
# Chosen: lc = 0.25 m
mesher = Mesher()
mesh_path = mesher.create(params_8_walls_angled, room_name=room_name, visualize=False, source_pos=[(2.5, 2.5, 1.5)])

directSimulator = DirectSimulator()
mic_positions = [
    (1, 1, 1.5),
    (2, 1, 1.5),
    (3, 1, 1.5),
    (2, 2, 1.5),
    (2, 1, 2.5),
    (1, 3, 1.5),
    (1.5, 1.3, 1.5),
    (2.2, 1.7, 1.5),
]

freqs, spl_responses = directSimulator.simulate(
    mesh_path,
    mic_positions=mic_positions,
    room_name=room_name,
)

evaluator = Evaluator()
msfd = evaluator.evaluate_msfd(
                response=spl_responses,
                input_is_db=True,
                weight_magnitude=0.5,
                weight_spatial=0.5,
            )["MSFD"]

print("MSFD: ", msfd)