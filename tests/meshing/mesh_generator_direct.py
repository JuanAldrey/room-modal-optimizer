from room_modal_optimizer.meshing.mesher import Mesher

rectangular_params = {
    "Lx": 4, "Ly": 3, "Lz": 3,

    "left_y0": 0, "left_y1": 0,
    "right_y0": 0, "right_y1": 0,
    "front_x0": 0, "front_x1": 0,
    "back_x0": 0, "back_x1": 0,

    "left_angle": 0,
    "right_angle": 0,
    "front_angle": 0,
    "back_angle": 0,
}

standard_params = {
    # Plant lengths
    "Lx": 4,
    "Ly": 3,
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
    "front_angle": 5,
    "back_angle": -10
}

offset_params = {
    "Lx": 4, "Ly": 3, "Lz": 3,

    "left_y0": 0.5, "left_y1": -0.3,
    "right_y0": -0.4, "right_y1": 0.4,
    "front_x0": 0.4, "front_x1": -0.3,
    "back_x0": -0.2, "back_x1": 0.5,

    "left_angle": 0,
    "right_angle": 0,
    "front_angle": 0,
    "back_angle": 0,
}

angle_params = {
    "Lx": 4, "Ly": 3, "Lz": 3,

    "left_y0": 0, "left_y1": 0,
    "right_y0": 0, "right_y1": 0,
    "front_x0": 0, "front_x1": 0,
    "back_x0": 0, "back_x1": 0,

    "left_angle": 12,
    "right_angle": -8,
    "front_angle": 7,
    "back_angle": -10,
}

aggresive_params = {
    "Lx": 4, "Ly": 3, "Lz": 3,

    "left_y0": 0.6, "left_y1": -0.4,
    "right_y0": -0.5, "right_y1": 0.5,
    "front_x0": 0.5, "front_x1": -0.4,
    "back_x0": -0.3, "back_x1": 0.6,

    "left_angle": 15,
    "right_angle": -15,
    "front_angle": 12,
    "back_angle": -12,
}

mesher = Mesher()
mesher.create(rectangular_params, room_name="rectangular", visualize=True, source_pos=(2, 1.5, 1.5))
mesher.create(standard_params, room_name="standard", visualize=True, source_pos=(2, 1.5, 1.5))
mesher.create(offset_params, room_name="offset", visualize=True, source_pos=(2, 1.5, 1.5))
mesher.create(angle_params, room_name="angle", visualize=True, source_pos=(2, 1.5, 1.5))
mesher.create(aggresive_params, room_name="aggressive", visualize=True, source_pos=(2, 1.5, 1.5))
