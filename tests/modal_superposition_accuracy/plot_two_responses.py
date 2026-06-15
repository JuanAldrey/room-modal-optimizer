from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Configuración
# =============================================================================

FILE_A = Path(
    "tests/modal_superposition_accuracy/results/direct_1/"
    "single_C001_direct.npz"
)

FILE_B = Path(
    "tests/modal_superposition_accuracy/results/modal_0.03/"
    "single_C001_modal.npz"
)

LABEL_A = "Directo"
LABEL_B = "Modal"

OUTPUT_DIR = Path(
    "tests/modal_superposition_accuracy/results/plots"
)


# =============================================================================
# Funciones
# =============================================================================

def loadSimulation(npzPath):
    data = np.load(npzPath)

    freqs = data["freqs"]

    if "spl_responses" in data:
        splResponses = data["spl_responses"]
    elif "H_modal" in data:
        hModal = data["H_modal"]
        splResponses = 20 * np.log10(np.abs(hModal) + 1e-12)
    else:
        raise ValueError(
            f"No encontré 'spl_responses' ni 'H_modal' en {npzPath}"
        )

    splResponses = ensureReceiversByFreqs(splResponses, freqs)

    return freqs, splResponses


def ensureReceiversByFreqs(splResponses, freqs):
    """
    Devuelve siempre shape:
        [n_mics, n_freqs]

    Directo suele venir:
        [n_freqs, n_mics]

    Modal suele venir:
        [n_mics, n_freqs]
    """
    splResponses = np.asarray(splResponses)

    if splResponses.ndim == 1:
        return splResponses.reshape(1, -1)

    if splResponses.shape[1] == len(freqs):
        return splResponses

    if splResponses.shape[0] == len(freqs):
        return splResponses.T

    raise ValueError(
        f"No puedo interpretar shape {splResponses.shape} "
        f"con {len(freqs)} frecuencias"
    )


def plotAverageResponse(freqsA, splA, freqsB, splB, labelA, labelB, outputPath):
    meanA = np.mean(splA, axis=0)
    meanB = np.mean(splB, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(freqsA, meanA, label=labelA)
    plt.plot(freqsB, meanB, label=labelB)

    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Nivel [dB]")
    plt.title("Respuesta promedio entre micrófonos")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outputPath, dpi=200)
    plt.close()


def plotMicResponses(freqsA, splA, freqsB, splB, labelA, labelB, outputDir):
    nMics = min(splA.shape[0], splB.shape[0])

    outputDir.mkdir(parents=True, exist_ok=True)

    for micIndex in range(nMics):
        plt.figure(figsize=(10, 5))

        plt.plot(freqsA, splA[micIndex], label=labelA)
        plt.plot(freqsB, splB[micIndex], label=labelB)

        plt.xlabel("Frecuencia [Hz]")
        plt.ylabel("Nivel [dB]")
        plt.title(f"Respuesta en frecuencia - Mic {micIndex + 1}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        outputPath = outputDir / f"mic_{micIndex + 1}_response.png"
        plt.savefig(outputPath, dpi=200)
        plt.close()


def plotAllMicsTogether(freqsA, splA, freqsB, splB, labelA, labelB, outputPath):
    nMics = min(splA.shape[0], splB.shape[0])

    plt.figure(figsize=(11, 6))

    for micIndex in range(nMics):
        plt.plot(
            freqsA,
            splA[micIndex],
            alpha=0.45,
            label=f"{labelA} M{micIndex + 1}",
        )
        plt.plot(
            freqsB,
            splB[micIndex],
            alpha=0.45,
            linestyle="--",
            label=f"{labelB} M{micIndex + 1}",
        )

    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Nivel [dB]")
    plt.title("Respuestas por micrófono superpuestas")
    plt.grid(True)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outputPath, dpi=200)
    plt.close()


def main():
    freqsA, splA = loadSimulation(FILE_A)
    freqsB, splB = loadSimulation(FILE_B)

    print("Simulación A:")
    print(f"  path: {FILE_A}")
    print(f"  label: {LABEL_A}")
    print(f"  freqs: {len(freqsA)} puntos, {freqsA[0]} Hz a {freqsA[-1]} Hz")
    print(f"  spl shape: {splA.shape}")

    print("Simulación B:")
    print(f"  path: {FILE_B}")
    print(f"  label: {LABEL_B}")
    print(f"  freqs: {len(freqsB)} puntos, {freqsB[0]} Hz a {freqsB[-1]} Hz")
    print(f"  spl shape: {splB.shape}")

    caseName = FILE_A.stem.replace("_direct", "")

    plotAverageResponse(
        freqsA=freqsA,
        splA=splA,
        freqsB=freqsB,
        splB=splB,
        labelA=LABEL_A,
        labelB=LABEL_B,
        outputPath=OUTPUT_DIR / f"{caseName}_average_response.png",
    )

    plotAllMicsTogether(
        freqsA=freqsA,
        splA=splA,
        freqsB=freqsB,
        splB=splB,
        labelA=LABEL_A,
        labelB=LABEL_B,
        outputPath=OUTPUT_DIR / f"{caseName}_all_mics_response.png",
    )

    plotMicResponses(
        freqsA=freqsA,
        splA=splA,
        freqsB=freqsB,
        splB=splB,
        labelA=LABEL_A,
        labelB=LABEL_B,
        outputDir=OUTPUT_DIR / f"{caseName}_mics",
    )

    print(f"Plots guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()