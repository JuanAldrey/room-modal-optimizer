from room_modal_optimizer.optimization.optimizer import Optimizer

gene_space_config = {
    "vertices": {
        "V1": {"dx": [-0.20, 0.20], "dy": [-0.20, 0.20]},
        "V2": {"dx": [-0.20, 0.20], "dy": [-0.20, 0.20]},
        "V3": {"dx": [-0.20, 0.20], "dy": [-0.20, 0.20]},
        "V4": {"dx": [-0.20, 0.20], "dy": [-0.20, 0.20]},
    },
    "walls": {},
    "Z": {"low": 3.0, "high": 4.2}
}

base_params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [0.0, 4.0],
            "V3": [5.0, 4.0],
            "V4": [5.0, 0.0],
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0,
        },
        "audience_area": {
            "V1": [1.6, 1.1],
            "V2": [1.6, 2.3],
            "V3": [3.4, 2.3],
            "V4": [3.4, 1.1],
        },
        "Z": 3.0,
        "source_pos": [2.5, 3.2, 1.5],
    }
}

optimizer = Optimizer(base_params=base_params, gene_space_config=gene_space_config)
optimizer.run()
optimizer.get_history()