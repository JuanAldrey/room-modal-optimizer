import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator


# =============================================================================
# Configuración
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

ROOMS_JSON = RESULTS_DIR / "single_square_fixed_source_many_mics.json"

ROOM_KEY = "single"
CONFIG_KEY = "C001"

RUN_NAME = f"{ROOM_KEY}_{CONFIG_KEY}_same_mesh_surface_source"

OUTPUT_DIR = RESULTS_DIR / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQS = np.arange(20.0, 201.0, 1.0)

FEM_ORDER = 2

N_MODES = 150
TARGET_FREQ = 100.0
MODAL_TOL = 1e-8

# Para comparación contra directo rígido sin impedancia:
ZETA = 0.00

SOURCE_STRENGTH = 0.01


# =============================================================================
# JSON helpers
# =============================================================================

def loadJson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cleanRoomParams(roomParams):
    data = roomParams["data"]

    return {
        "data": {
            "vertices": data["vertices"],
            "walls": data["walls"],
            "Z": data["Z"],
        }
    }


def getRoomSource(roomParams):
    data = roomParams["data"]

    if "source" not in data:
        raise KeyError("No encontré data['source']. Este script espera fuente fija.")

    return tuple(data["source"])


def sortedMicPositions(config):
    mics = config["mics"]

    micKeys = sorted(
        mics.keys(),
        key=lambda key: int(key[1:])
    )

    return [
        tuple(mics[micKey])
        for micKey in micKeys
    ]


def loadSingleCase():
    experimentRooms = loadJson(ROOMS_JSON)

    if ROOM_KEY not in experimentRooms:
        raise KeyError(f"No existe ROOM_KEY='{ROOM_KEY}' en {ROOMS_JSON}")

    roomParams = experimentRooms[ROOM_KEY]
    positionConfigs = roomParams["data"]["position_configs"]

    if CONFIG_KEY not in positionConfigs:
        raise KeyError(f"No existe CONFIG_KEY='{CONFIG_KEY}' en room '{ROOM_KEY}'")

    config = positionConfigs[CONFIG_KEY]

    sourcePosition = getRoomSource(roomParams)
    micPositions = sortedMicPositions(config)
    cleanParams = cleanRoomParams(roomParams)

    return cleanParams, sourcePosition, micPositions


# =============================================================================
# Helpers numéricos
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
    plt.plot(freqs, modalMean, label="Modal rígido - fuente superficial fija")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Nivel [dB]")
    plt.title(f"Respuesta promedio - {ROOM_KEY} {CONFIG_KEY}")
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
    plt.title(f"Respuesta promedio centrada - {ROOM_KEY} {CONFIG_KEY}")
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
    plt.title(f"Respuestas centradas por micrófono - {ROOM_KEY} {CONFIG_KEY}")
    plt.grid(True)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=200)
    plt.close()


# =============================================================================
# Simulaciones
# =============================================================================

def createSharedMesh(roomParams, sourcePosition):
    print()
    print("=" * 80)
    print("Creando malla compartida con Source fija")
    print("=" * 80)

    mesher = Mesher()

    meshPath = mesher.create(
        roomParams,
        room_name=RUN_NAME,
        visualize=False,
        source_pos=sourcePosition,
    )

    print(f"Shared mesh path: {meshPath}")
    print(f"Source position: {sourcePosition}")

    return meshPath


def runDirect(meshPath, micPositions):
    print()
    print("=" * 80)
    print("Running direct rigid")
    print("=" * 80)

    directSimulator = DirectSimulator()

    directSimulator.freqs = FREQS

    freqs, splResponses = directSimulator.simulate(
        meshPath,
        mic_positions=micPositions,
        room_name=f"{RUN_NAME}_direct",
        export=False,
    )

    splResponses = ensureMicsByFreqs(splResponses, freqs)

    return np.asarray(freqs, dtype=float), splResponses


def runModal(meshPath, micPositions):
    print()
    print("=" * 80)
    print("Running modal rigid - same mesh / fixed surface source")
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
        tol=MODAL_TOL,
    )

    modalSimulator.obtainModes()
    modalSimulator.sortModes()

    eigFreq = np.asarray(modalSimulator.eig_freq, dtype=float)

    if len(eigFreq) == 0:
        raise RuntimeError("No convergió ningún modo.")

    print()
    print("Modos:")
    print(f"  n modes converged: {len(eigFreq)}")
    print(f"  min mode: {np.min(eigFreq):.6f} Hz")
    print(f"  max mode: {np.max(eigFreq):.6f} Hz")
    print(f"  first modes: {eigFreq[:10]}")
    print(f"  last modes: {eigFreq[-10:]}")

    sourceWeights = modalSimulator.computeSourceSurfaceWeights()

    H = modalSimulator.modalTransferFromFixedSurfaceSource(
        receiverPositions=micPositions,
        freqs=FREQS,
        sourceWeights=sourceWeights,
        zeta=ZETA,
        sourceStrength=SOURCE_STRENGTH,
    )

    modalSpl = 20.0 * np.log10(np.abs(H) + 1e-12)

    return np.asarray(FREQS, dtype=float), modalSpl, eigFreq, sourceWeights


# =============================================================================
# Main
# =============================================================================

def main():
    roomParams, sourcePosition, micPositions = loadSingleCase()

    print()
    print("=" * 80)
    print("Single case modal superposition accuracy")
    print("=" * 80)
    print(f"ROOM_KEY: {ROOM_KEY}")
    print(f"CONFIG_KEY: {CONFIG_KEY}")
    print(f"SOURCE: {sourcePosition}")
    print(f"MICS: {micPositions}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    meshPath = createSharedMesh(
        roomParams=roomParams,
        sourcePosition=sourcePosition,
    )

    directFreqs, directSpl = runDirect(
        meshPath=meshPath,
        micPositions=micPositions,
    )

    modalFreqs, modalSpl, eigFreq, sourceWeights = runModal(
        meshPath=meshPath,
        micPositions=micPositions,
    )

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
    print("Resultados comparación")
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
        OUTPUT_DIR / f"{RUN_NAME}_results.npz",
        freqs=np.asarray(directFreqs),
        direct_spl=np.asarray(directSpl),
        modal_spl=np.asarray(modalSpl),
        eig_freq=np.asarray(eigFreq),
        source_weights=np.asarray(sourceWeights),
        source_position=np.asarray(sourcePosition),
        mic_positions=np.asarray(micPositions),
        room_key=np.asarray(ROOM_KEY),
        config_key=np.asarray(CONFIG_KEY),
        fem_order=np.asarray(FEM_ORDER),
        n_modes=np.asarray(N_MODES),
        target_freq=np.asarray(TARGET_FREQ),
        zeta=np.asarray(ZETA),
        source_strength=np.asarray(SOURCE_STRENGTH),
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