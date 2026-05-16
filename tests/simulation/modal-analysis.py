from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator

room_name = 'testing_non_rectangular_6_4_3_4'
params = {
    # Plant lengths
    "Lx": 6,
    "Ly": 4,
    "Lz": 3,

    # Plant offsets
    "left_y0": 0,
    "left_y1": 0,
    "right_y0": 0,
    "right_y1": 0,
    "front_x0": 0,
    "front_x1": 0,
    "back_x0": 0,
    "back_x1": 0,

    # Wall inclination (degrees)
    "left_angle": 0,
    "right_angle": 0,
    "front_angle": 0,
    "back_angle": 0
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