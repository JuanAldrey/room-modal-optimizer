from room_modal_optimizer.optimization.absorption_optimizer import AbsorptionOptimizer

params = {
    "data": {
        "vertices": {
            "V1": [0.0, 0.0],
            "V2": [5.0, 0.0],
            "V3": [5.0, 4.0],
            "V4": [0.0, 4.0],
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0,
        },
        "Z": 3.0,
        "source_pos": [[2.4, 3.7, 1.5], [3.6, 3.7, 1.5]],
    }
}

mic_positions = [
    [1.3, 0.9, 1.2],
    [3.0, 0.9, 1.2],
    [4.7, 0.9, 1.2],
]

absorptionOptimizer = AbsorptionOptimizer(percentage=0.2, params=params, mic_positions=mic_positions)
best_results = absorptionOptimizer.run()

print(best_results)