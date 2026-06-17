from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
import matplotlib.pyplot as plt

room_name = 'testing_rectangular_5_3_4'
params = {
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

# lc chosen from highest frequency:
# lambda_min = c / f_max = 343 / 200 = 1.715 m
# Use ~6 elems per wavelength:
# lc = 1.715 / 6 = 0.286 m
# Chosen: lc = 0.25 m
mesher = Mesher()
mesh_path = mesher.create(params, room_name=room_name, visualize=False, source_pos=(2.5, 2.5, 1.5))

directSimulator = DirectSimulator()
mic_positions = [
    (1, 1, 1.5),
    (2, 1, 1.5),
    (3, 1, 1.5),
    (2, 2, 1.5),
    (2, 1, 2.5),
    (1, 3, 1.5),
    (1.5, 1.3, 1.5),
    (2.2, 1.7, 1.5),
]

freqs, spl_responses = directSimulator.simulate(
    mesh_path,
    mic_positions=mic_positions,
    room_name=room_name,
)

labels = [
    f"Mic {i + 1} ({x}, {y}, {z})"
    for i, (x, y, z) in enumerate(mic_positions)
]

plt.figure()

for m in range(spl_responses.shape[1]):
    plt.plot(freqs, spl_responses[:, m], label=labels[m])

plt.xlabel("Frequency [Hz]")
plt.ylabel("SPL [dB]")
plt.title("SPL response")
plt.grid(True)
plt.legend()
plt.show()



