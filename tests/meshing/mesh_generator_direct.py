from room_modal_optimizer.meshing.mesher import Mesher

room_name = "nice_mesh_room"

params = {
    "data": {
        "vertices": {
            "V1": [-3.00, 0.00],
            "V2": [ 3.00, 0.00],
            "V3": [ 3.40, 2.20],
            "V4": [ 2.10, 5.20],
            "V5": [-2.10, 5.20],
            "V6": [-3.40, 2.20],
        },

        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0,
            "W5": 0.0,
            "W6": 0.0,
        },

        "audience_area": {
            "V1": [-1.60, 2.00],
            "V2": [ 1.60, 2.00],
            "V3": [ 1.30, 4.20],
            "V4": [-1.30, 4.20],
        },

        "Z": 3.40,

        "source_pos": [[0.00, 1.00, 1.20]],
    }
}

mesher = Mesher()
mesher.create(params, room_name="rectangular", visualize=True, source_pos=[(2, 1.5, 1.5)])
