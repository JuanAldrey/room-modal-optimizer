from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator

room_name = 'testing_non_rectangular_6_4_3_4'
params = {
    # Plant lengths
    "Lx": 5,
    "Ly": 3,
    "Lz": 3,

    # Plant offsets
    "left_y0": 0,
    "left_y1": 0,
    "right_y0": 0,
    "right_y1": 0,
    "front_x0": 0.2,
    "front_x1": -0.5,
    "back_x0": 0.6,
    "back_x1": 0.8,

    # Wall inclination (degrees)
    "left_angle": 5,
    "right_angle": -5,
    "front_angle": -10,
    "back_angle": 5
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

evaluator = Evaluator()
frq_sp_idx = evaluator.evaluate(eig_freq, n_modes)

print(eig_freq)
print("Index: ", frq_sp_idx)