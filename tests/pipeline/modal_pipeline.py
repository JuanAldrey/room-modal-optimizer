from room_modal_optimizer.pipeline.modal_pipeline import ModalPipeline

room_name = 'testing_pipeline'
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

modalPipeline = ModalPipeline()
idx = modalPipeline.run(params, room_name=room_name)

print("Fitness result: ", idx)