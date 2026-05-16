from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator

room_name = 'testing_non_rectangular_6_4_3_3'
params = {
    # Plant lengths
    "Lx": 6,
    "Ly": 4,
    "Lz": 3,

    # Plant offsets
    "left_y0": 0.5,
    "left_y1": -0.5,
    "right_y0": -0.4,
    "right_y1": 0.4,
    "front_x0": 0.6,
    "front_x1": -0.6,
    "back_x0": 0.7,
    "back_x1": 0.7,

    # Wall inclination (degrees)
    "left_angle": 6,
    "right_angle": 8,
    "front_angle": 7,
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
eig_freq, eig_vector = modalSimulator.simulate(mesh_path, room_name=room_name, export=False)

print(eig_freq)