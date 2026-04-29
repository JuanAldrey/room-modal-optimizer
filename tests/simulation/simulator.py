from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator

room_name = 'testing_rectangular_4_3_3'
params = {
    # Plant lengths
    "Lx": 4,
    "Ly": 3,
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
mesh_path = mesher.create(params, lc=0.25, room_name=room_name, visualize=True)

modalSimulator = ModalSimulator()
modalSimulator.simulate(mesh_path, room_name=room_name, export=True)

directSimulator = DirectSimulator()
directSimulator.simulate(mesh_path, source_position=(2.0,2.0,1.5), mic_positions=[1, 1, 1.5], room_name=room_name)

