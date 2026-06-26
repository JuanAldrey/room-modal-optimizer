from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
import matplotlib.pyplot as plt
room_name = "testing_patched"

params_8_walls_angled = {
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
            "W1": 3.0,
            "W2": 0.0,
            "W3": 0.3,
            "W4": 5.0,
            "W5": 0.0,
            "W6": 5.0,
            "W7": -3.0,
            "W8": -3.0
        },
        "Z": 3.0
    }
}

mesher = Mesher()
mesh_path, phyisical_tags, patchedArea = mesher.create(params_8_walls_angled, room_name="rectangular", visualize=False, source_pos=[[2, 2.5, 1.5]], patch=True)

def createPatchImpedanceByTag(physicalTags, patchImpedance=25.0):
    return {
        tag: patchImpedance
        for name, tag in physicalTags.items()
        if (
            name.startswith("CeilingPatch_")
            or name.startswith("WallPatch_")
        )
    }

patchImpedanceByTag = createPatchImpedanceByTag(
    phyisical_tags,
    patchImpedance=25.0
)

directSimulator = DirectSimulator()
mic_positions = [
    (1, 1, 1.5),
    (2, 1, 1.5),
    (3, 1, 1.5),
    (2, 2, 1.5),
    (2, 1, 2.5),
    (1, 3, 1.5),
    (1.5, 1.3, 1.5),
    (2.2, 1.7, 1.5)
]

freqs, spl_responses = directSimulator.simulate(
    mesh_path,
    mic_positions=mic_positions,
    room_name=room_name,
    patch=True,
    impedance_mappings=patchImpedanceByTag
)

labels = [
    f"Mic {i + 1} ({x}, {y}, {z})"
    for i, (x, y, z) in enumerate(mic_positions)
]

plt.figure()

for m in range(spl_responses.shape[0]):
    plt.plot(freqs, spl_responses[m, :], label=labels[m])

plt.xlabel("Frequency [Hz]")
plt.ylabel("SPL [dB]")
plt.title("SPL response")
plt.grid(True)
plt.legend()
plt.show()



