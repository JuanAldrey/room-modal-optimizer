from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
import matplotlib.pyplot as plt

room_name = 'testing_rectangular_5_3_4'
params = {
    # Plant lengths
    "Lx": 5,
    "Ly": 3,
    "Lz": 4,

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
mesh_path = mesher.create(params, room_name=room_name, visualize=False, source_pos=(2.5, 2.5, 1.5))

directSimulator = DirectSimulator()
freqs, spl_responses = directSimulator.simulate(
    mesh_path, 
    mic_positions = [
        (1, 1, 1.5),
        (2, 1, 1.5),
    ],
    room_name=room_name,
    export=True
    )

labels = ["Mic (1,1,1.5)", "Mic (2,1,1.5)"]

plt.figure()
for m in range(spl_responses.shape[1]):
    plt.plot(freqs, spl_responses[:, m], label=labels[m])

plt.xlabel("Frequency [Hz]")
plt.ylabel("SPL [dB]")
plt.title("SPL response")
plt.grid(True)
plt.legend()
plt.show()



