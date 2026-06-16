from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
isRoot = rank == 0


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
if isRoot:
    mesh_path = mesher.create(
        params,
        room_name=room_name,
        visualize=False,
        source_pos=(2.5, 2.5, 1.5)
    )
else:
    mesh_path = None

mesh_path = comm.bcast(mesh_path, root=0)

directSimulator = DirectSimulator()
freqs, spl_responses = directSimulator.simulate(
    mesh_path, 
    mic_positions = [
        (1, 1, 1.5),
        (2, 1, 1.5),
    ],
    room_name=room_name
    )

if spl_responses is not None:

    print("freqs shape:", freqs.shape)
    print("spl shape:", spl_responses.shape)
    print("has -inf:", np.isneginf(spl_responses).any())
    print("has nan:", np.isnan(spl_responses).any())
    print("min spl:", np.nanmin(spl_responses))
    print("max spl:", np.nanmax(spl_responses))

    labels = ["Mic (1,1,1.5)", "Mic (2,1,1.5)"]

    plt.figure()

    for m in range(spl_responses.shape[0]):
        label = labels[m] if m < len(labels) else f"Mic {m + 1}"
        plt.plot(freqs, spl_responses[m, :], label=label)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("SPL [dB]")
    plt.title("SPL response")
    plt.grid(True)
    plt.legend()
    plt.show()


