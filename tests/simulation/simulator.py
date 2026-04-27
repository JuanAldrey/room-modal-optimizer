from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.simulator import Simulator

params = {
    # Plant lengths
    "Lx": 4,
    "Ly": 4,
    "Lz": 3,

    # Plant offsets
    "left_y0": 0.1,
    "left_y1": 0.2,
    "right_y0": -0.2,
    "right_y1": -0.3,
    "front_x0": 0.4,
    "front_x1": 0,
    "back_x0": 0.2,
    "back_x1": -0.3,

    # Wall inclination (degrees)
    "left_angle": 10,
    "right_angle": -10,
    "front_angle": 10,
    "back_angle": -10
}

mesher = Mesher(params, lc=0.25)
mesh_path = mesher.create(visualize=True)

simulator = Simulator()
simulator.simulate(mesh_path, source_position=(2.0,2.0,1.5), mic_positions=[1, 1, 1.5])