from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator


# =============================================================================
# Configuración
# =============================================================================

ROOM_NAME = "rigid_same_mesh_surface_source"

OUTPUT_DIR = Path(
    "tests/modal_superposition_accuracy/results/rigid_same_mesh_surface_source_raw"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQS = np.arange(20.0, 101.0, 2.0)

# Usá el mismo orden que tu DirectSimulator.
# Si DirectSimulator usa ("Lagrange", 1), dejá 1.
# Si usa ("Lagrange", 2), cambiá esto a 2.
FEM_ORDER = 1

N_MODES = 300
TARGET_FREQ = 100.0
ZETA = 0.0

SOURCE_STRENGTH = 0.01

SOURCE_POSITION = [1.0, 0.8, 1.2]

MIC_POSITIONS = [
    [2.0, 2.0, 1.2],
    [2.5, 2.0, 1.2],
    [3.0, 2.0, 1.2],
]

ROOM_PARAMS = {
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
    }
}


# =============================================================================
# Helpers
# =============================================================================

def ensureMicsByFreqs(splResponses, freqs):
    """
    Devuelve siempre shape:
        [n_mics, n_freqs]
    """
    splResponses = np.asarray(splResponses)
    freqs = np.asarray(freqs)

    if splResponses.ndim != 2:
        raise ValueError(
            f"splResponses debe ser 2D. Shape recibido: {splResponses.shape}"
        )

    if splResponses.shape[0] == len(freqs):
        return splResponses.T

    if splResponses.shape[1] == len(freqs):
        return splResponses

    raise ValueError(
        f"No puedo interpretar shape={splResponses.shape} "
        f"con len(freqs)={len(freqs)}"
    )


def sortModes(modalSimulator):
    pairs = sorted(
        zip(modalSimulator.eig_freq, modalSimulator.eig_vector),
        key=lambda pair: pair[0],
    )

    modalSimulator.eig_freq = [freq for freq, vec in pairs]
    modalSimulator.eig_vector = [vec for freq, vec in pairs]


def centerCurve(curve):
    curve = np.asarray(curve, dtype=float)
    return curve - np.mean(curve)


def correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


# =============================================================================
# Plots
# =============================================================================

def saveMeanRawPlot(freqs, directSpl, modalSpl, outputPath):
    directMean = np.mean(directSpl, axis=0)
    modalMean = np.mean(modalSpl, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, directMean, label="Directo rígido")
    plt.plot(freqs, modalMean, label="Modal rígido - fuente superficial")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Nivel [dB]")
    plt.title("Respuesta promedio entre micrófonos")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputPath, dpi=200)
    plt.close()


def saveMeanCenteredPlot(freqs, directSpl, modalSpl, outputPath):
    directMean = np.mean(directSpl, axis=0)
    modalMean = np.mean(modalSpl, axis=0)

    directCentered = centerCurve(directMean)
    modalCentered = centerCurve(modalMean)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, directCentered, label="Directo rígido centrado")
    plt.plot(freqs, modalCentered, label="Modal rígido centrado")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Nivel relativo [dB]")
    plt.title("Respuesta promedio centrada")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputPath, dpi=200)
    plt.close()


def saveMicsCenteredPlot(freqs, directSpl, modalSpl, outputPath):
    nMics = min(directSpl.shape[0], modalSpl.shape[0])

    plt.figure(figsize=(11, 6))

    for micIndex in range(nMics):
        directCentered = centerCurve(directSpl[micIndex])
        modalCentered = centerCurve(modalSpl[micIndex])

        plt.plot(
            freqs,
            directCentered,
            label=f"Directo M{micIndex + 1}",
            alpha=0.75,
        )

        plt.plot(
            freqs,
            modalCentered,
            linestyle="--",
            label=f"Modal M{micIndex + 1}",
            alpha=0.75,
        )

    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Nivel relativo [dB]")
    plt.title("Respuestas centradas por micrófono")
    plt.grid(True)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=200)
    plt.close()


# =============================================================================
# Simulaciones
# =============================================================================

def createSharedMesh():
    print()
    print("=" * 80)
    print("Creando malla compartida con Source")
    print("=" * 80)

    mesher = Mesher()

    meshPath = mesher.create(
        ROOM_PARAMS,
        room_name=ROOM_NAME,
        visualize=False,
        source_pos=SOURCE_POSITION,
    )

    print(f"Shared mesh path: {meshPath}")

    return meshPath


def runDirect(meshPath):
    print()
    print("=" * 80)
    print("Running direct rigid")
    print("=" * 80)

    directSimulator = DirectSimulator()

    directSimulator.freqs = FREQS

    freqs, splResponses = directSimulator.simulate(
        mesh_path=meshPath,
        mic_positions=MIC_POSITIONS,
        room_name=f"{ROOM_NAME}_direct",
        export=False,
    )

    splResponses = ensureMicsByFreqs(splResponses, freqs)

    return np.asarray(freqs, dtype=float), splResponses


def runModal(meshPath):
    print()
    print("=" * 80)
    print("Running modal rigid - same mesh / surface source")
    print("=" * 80)

    print(f"FEM_ORDER: {FEM_ORDER}")
    print(f"N_MODES requested: {N_MODES}")
    print(f"TARGET_FREQ requested: {TARGET_FREQ}")
    print(f"ZETA: {ZETA}")

    modalSimulator = ModalSimulator()

    modalSimulator.loadMesh(meshPath)
    modalSimulator.setup(order=FEM_ORDER)

    modalSimulator.computeModalAnalysis(
        target_freq=TARGET_FREQ,
        n_modes=N_MODES,
        tol=1e-8,
    )

    modalSimulator.obtainModes()
    sortModes(modalSimulator)

    eigFreq = np.asarray(modalSimulator.eig_freq, dtype=float)

    print()
    print("Modos:")
    print(f"  n modes converged: {len(eigFreq)}")
    print(f"  min mode: {np.min(eigFreq):.6f} Hz")
    print(f"  max mode: {np.max(eigFreq):.6f} Hz")
    print(f"  first modes: {eigFreq[:10]}")
    print(f"  last modes: {eigFreq[-10:]}")

    H = modalSimulator.modalTransferFromSurfaceSource(
        receiverPositions=MIC_POSITIONS,
        freqs=FREQS,
        zeta=ZETA,
        sourceStrength=SOURCE_STRENGTH,
    )

    modalSpl = 20.0 * np.log10(np.abs(H) + 1e-12)

    return np.asarray(FREQS, dtype=float), modalSpl, eigFreq


# =============================================================================
# Main
# =============================================================================

def main():
    meshPath = createSharedMesh()

    directFreqs, directSpl = runDirect(meshPath)
    modalFreqs, modalSpl, eigFreq = runModal(meshPath)

    if not np.allclose(directFreqs, modalFreqs):
        raise ValueError("Las frecuencias de directo y modal no coinciden.")

    directMean = np.mean(directSpl, axis=0)
    modalMean = np.mean(modalSpl, axis=0)

    directCentered = centerCurve(directMean)
    modalCentered = centerCurve(modalMean)

    meanCorr = correlation(directCentered, modalCentered)
    meanCenteredMae = np.mean(np.abs(directCentered - modalCentered))

    print()
    print("=" * 80)
    print("Resultados comparación raw")
    print("=" * 80)
    print(f"Mesh compartida: {meshPath}")
    print(f"Direct SPL shape: {directSpl.shape}")
    print(f"Modal SPL shape: {modalSpl.shape}")

    print()
    print("Promedio entre mics:")
    print(f"  Mean centered correlation: {meanCorr:.3f}")
    print(f"  Mean centered MAE: {meanCenteredMae:.3f} dB")

    print()
    print("Por micrófono:")

    for micIndex in range(directSpl.shape[0]):
        directMicCentered = centerCurve(directSpl[micIndex])
        modalMicCentered = centerCurve(modalSpl[micIndex])

        micCorr = correlation(directMicCentered, modalMicCentered)
        micMae = np.mean(np.abs(directMicCentered - modalMicCentered))

        print(
            f"  M{micIndex + 1}: "
            f"corr={micCorr:.3f}, MAE={micMae:.3f} dB"
        )

    np.savez_compressed(
        OUTPUT_DIR / "rigid_same_mesh_surface_source_raw_results.npz",
        freqs=np.asarray(directFreqs),
        direct_spl=np.asarray(directSpl),
        modal_spl=np.asarray(modalSpl),
        eig_freq=np.asarray(eigFreq),
        source_position=np.asarray(SOURCE_POSITION),
        mic_positions=np.asarray(MIC_POSITIONS),
    )

    saveMeanRawPlot(
        freqs=directFreqs,
        directSpl=directSpl,
        modalSpl=modalSpl,
        outputPath=OUTPUT_DIR / "mean_raw.png",
    )

    saveMeanCenteredPlot(
        freqs=directFreqs,
        directSpl=directSpl,
        modalSpl=modalSpl,
        outputPath=OUTPUT_DIR / "mean_centered.png",
    )

    saveMicsCenteredPlot(
        freqs=directFreqs,
        directSpl=directSpl,
        modalSpl=modalSpl,
        outputPath=OUTPUT_DIR / "mics_centered.png",
    )

    print()
    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print()
    print("Archivos principales:")
    print(f"  {OUTPUT_DIR / 'mean_raw.png'}")
    print(f"  {OUTPUT_DIR / 'mean_centered.png'}")
    print(f"  {OUTPUT_DIR / 'mics_centered.png'}")


if __name__ == "__main__":
    main()