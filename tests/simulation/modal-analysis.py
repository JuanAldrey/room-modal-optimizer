from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator

room_name = 'testing_non_rectangular_6_4_3_4'
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

# lc chosen from highest frequency:
# lambda_min = c / f_max = 343 / 200 = 1.715 m
# Use ~6 elems per wavelength:
# lc = 1.715 / 6 = 0.286 m
# Chosen: lc = 0.25 m
mesher = Mesher()
mesh_path = mesher.create(params, lc=0.25, room_name=room_name, visualize=False)

modalSimulator = ModalSimulator()
eig_freq, eig_vector, n_modes = modalSimulator.simulate(mesh_path, room_name=room_name, export=False)

print(eig_freq)