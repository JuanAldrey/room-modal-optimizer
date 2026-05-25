from room_modal_optimizer.meshing.mesher import Mesher

rectangular_params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [4.0, 0.0],
            "V3": [4.0, 3.0],
            "V4": [0.0, 3.0]
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0
        },
        "Z": 3.0
    }
}

one_angle_params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [4.0, 0.0],
            "V3": [4.0, 3.0],
            "V4": [0.0, 3.0]
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 5.0,
            "W4": 0.0
        },
        "Z": 3.0
    }
}

irregular_params = {
    "data": {
        "vertices": {
            "V1": [1.182, -0.026],
            "V2": [0.804, 0.869],
            "V3": [-0.07, 0.68],
            "V4": [0.015, -0.003]
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 3.0,
            "W4": 0.0
        },
        "Z": 3.0
    }
}

params_5_walls = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [4.0, 0.0],
            "V3": [4.5, 1.5],
            "V4": [2.5, 3.0],
            "V5": [0.0, 2.5]
        },
        "walls": {
            "W1": 0.0,
            "W2": 5.0,
            "W3": 0.0,
            "W4": -3.0,
            "W5": 0.0
        },
        "Z": 3.0
    }
}

params_8_walls = {
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
            "W2": 3.0,
            "W3": -4.0,
            "W4": 5.0,
            "W5": 0.0,
            "W6": -3.0,
            "W7": 2.0,
            "W8": 0.0
        },
        "Z": 3.0
    }
}

params_8_walls_no_angles = {
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

params_clockwise = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [0.0, 3.0],
            "V3": [4.0, 3.0],
            "V4": [4.0, 0.0]
        },
        "walls": {
            "W1": 0.0,
            "W2": 5.0,
            "W3": 0.0,
            "W4": -5.0
        },
        "Z": 3.0
    }
}

aggressive_params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [2.0, -0.2],
            "V3": [4.0, 0.3],
            "V4": [4.7, 1.6],
            "V5": [4.0, 3.0],
            "V6": [2.4, 3.5],
            "V7": [0.7, 3.0],
            "V8": [-0.4, 1.4],
        },
        "walls": {
            "W1": 0.22775101461778569,
            "W2": 7.625839293322446,
            "W3": -7.256793396480036,
            "W4": 1.7207176304230138,
            "W5": -5.2716140210033355,
            "W6": -6.959174512235528,
            "W7": -1.4234077869082995,
            "W8": 7.45011252919295,
        },
        "Z": 3.3700768177397533,
    }
}

mesher = Mesher()
#mesher.create(rectangular_params, room_name="rectangular", visualize=True)
#mesher.create(one_angle_params, room_name="standard", visualize=True)
#mesher.create(irregular_params, room_name="offset", visualize=True)
#mesher.create(params_5_walls, room_name="angle", visualize=True)
#mesher.create(params_8_walls, room_name="aggressive", visualize=True)
#mesher.create(params_8_walls_no_angles, room_name="aggressive", visualize=True)
#mesher.create(params_clockwise, room_name="aggressive", visualize=True)
mesher.create(aggressive_params, room_name="aggressive", visualize=True)

