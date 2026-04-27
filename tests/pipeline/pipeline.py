from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.simulator import Simulator

params = {
    # Plant lengths
    "Lx": 10,
    "Ly": 10,
    "Lz": 3,

    # Plant offsets
    "left_y0": 0.3,
    "left_y1": 0.2,
    "right_y0": 0.0,
    "right_y1": -0.1,
    "front_x0": 0.3,
    "front_x1": 0.1,
    "back_x0": 0.2,
    "back_x1": -0.1,

    # Wall inclination (degrees)
    "left_angle": 10,
    "right_angle": -10,
    "front_angle": -20,
    "back_angle": -20
}

mesher = Mesher(params, lc=0.3)
mesher.create()