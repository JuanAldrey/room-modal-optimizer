import json
import time
import traceback
from pathlib import Path

import numpy as np

from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator


# =========================================================
# Paths
# =========================================================

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

ROOMS_JSON = RESULTS_DIR / "single_square_fixed_source_many_mics.json"

RUN_LABEL = "overnight_sweep"

OUT_DIR = RESULTS_DIR / RUN_LABEL
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "summary.json"


# =========================================================
# Sweep config
# =========================================================

FREQ_MIN = 20.0
FREQ_MAX = 201.0

FREQ_STEPS = [0.5, 1.0, 2.0]

USE_IMPEDANCE_VALUES = [False, True]

ZETA_VALUES = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015]

IMPEDANCE_VALUE = 25.0 + 0j

MODAL_ORDER = 2
N_MODES = 150
TARGET_FREQ = 100.0
MODAL_TOL = 1e-8

SOURCE_STRENGTH = 0.01

SKIP_COMPLETED = True


# =========================================================
# Helpers
# =========================================================

def loadJson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def saveJson(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmpPath = path.with_suffix(".tmp")

    with open(tmpPath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    tmpPath.replace(path)


def loadSummary():
    if SUMMARY_PATH.exists():
        return loadJson(SUMMARY_PATH)

    return {
        "status": "running",
        "run_label": RUN_LABEL,
        "freq_steps": FREQ_STEPS,
        "use_impedance_values": USE_IMPEDANCE_VALUES,
        "zeta_values": ZETA_VALUES,
        "modal_order": MODAL_ORDER,
        "n_modes": N_MODES,
        "target_freq": TARGET_FREQ,
        "cases": {},
        "rooms": {},
    }


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
    return tuple(roomParams["data"]["source"])


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


def freqsFromStep(freqStep):
    return np.arange(FREQ_MIN, FREQ_MAX + 0.5 * freqStep, freqStep)


def boolLabel(value):
    return "impedance" if value else "rigid"


def directCaseName(roomName, configName, freqStep, useImpedance):
    return (
        f"{roomName}_{configName}"
        f"_direct_{boolLabel(useImpedance)}"
        f"_step_{freqStep}"
    )


def modalCaseName(roomName, configName, freqStep, zeta):
    return (
        f"{roomName}_{configName}"
        f"_modal_zeta_{zeta}"
        f"_step_{freqStep}"
    )


def getDirectPath(caseName):
    return OUT_DIR / "direct" / f"{caseName}.npz"


def getModalPath(caseName):
    return OUT_DIR / "modal" / f"{caseName}.npz"


def getRoomMeshName(roomName):
    return f"{roomName}_{RUN_LABEL}_mesh"


# =========================================================
# Room setup
# =========================================================

def createRoomMesh(roomName, roomParams):
    cleanParams = cleanRoomParams(roomParams)
    sourcePos = getRoomSource(roomParams)

    mesher = Mesher()

    meshPath = mesher.create(
        cleanParams,
        room_name=getRoomMeshName(roomName),
        visualize=False,
        source_pos=sourcePos,
    )

    return meshPath, sourcePos


def setupModalRoom(meshPath):
    modalSimulator = ModalSimulator()

    modalSimulator.loadMesh(meshPath)
    modalSimulator.setup(order=MODAL_ORDER)

    modalSimulator.computeModalAnalysis(
        target_freq=TARGET_FREQ,
        n_modes=N_MODES,
        tol=MODAL_TOL,
    )

    modalSimulator.obtainModes()
    modalSimulator.sortModes()

    sourceWeights = modalSimulator.computeSourceSurfaceWeights()

    return modalSimulator, sourceWeights


# =========================================================
# Direct / Modal cases
# =========================================================

def runDirectCase(
    meshPath,
    sourcePos,
    roomName,
    configName,
    config,
    freqStep,
    useImpedance,
):
    caseName = directCaseName(
        roomName=roomName,
        configName=configName,
        freqStep=freqStep,
        useImpedance=useImpedance,
    )

    outputPath = getDirectPath(caseName)

    if SKIP_COMPLETED and outputPath.exists():
        print(f"Skipping direct: {caseName}")
        return str(outputPath)

    print(f"Running direct: {caseName}", flush=True)

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    freqs = freqsFromStep(freqStep)
    micPositions = sortedMicPositions(config)

    directSimulator = DirectSimulator()

    freqsOut, splResponses = directSimulator.simulate(
        mesh_path=meshPath,
        mic_positions=micPositions,
        room_name=caseName,
        freqs=freqs,
        use_impedance=useImpedance,
        wall_z=IMPEDANCE_VALUE,
        floor_z=IMPEDANCE_VALUE,
        ceiling_z=IMPEDANCE_VALUE,
    )

    np.savez_compressed(
        outputPath,
        freqs=np.asarray(freqsOut),
        spl_responses=np.asarray(splResponses),
        source_position=np.asarray(sourcePos),
        mic_positions=np.asarray(micPositions),
        use_impedance=np.asarray(useImpedance),
        impedance_value=np.asarray(IMPEDANCE_VALUE),
        freq_step=np.asarray(freqStep),
    )

    return str(outputPath)


def runModalCase(
    modalSimulator,
    sourceWeights,
    sourcePos,
    roomName,
    configName,
    config,
    freqStep,
    zeta,
):
    caseName = modalCaseName(
        roomName=roomName,
        configName=configName,
        freqStep=freqStep,
        zeta=zeta,
    )

    outputPath = getModalPath(caseName)

    if SKIP_COMPLETED and outputPath.exists():
        print(f"Skipping modal: {caseName}")
        return str(outputPath)

    print(f"Running modal: {caseName}", flush=True)

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    freqs = freqsFromStep(freqStep)
    micPositions = sortedMicPositions(config)

    H = modalSimulator.modalTransferFromFixedSurfaceSource(
        receiverPositions=micPositions,
        freqs=freqs,
        sourceWeights=sourceWeights,
        zeta=zeta,
        sourceStrength=SOURCE_STRENGTH,
    )

    splResponses = 20.0 * np.log10(np.abs(H) + 1e-12)

    np.savez_compressed(
        outputPath,
        freqs=np.asarray(freqs),
        H_modal=np.asarray(H),
        spl_responses=np.asarray(splResponses),
        source_position=np.asarray(sourcePos),
        mic_positions=np.asarray(micPositions),
        eig_freq=np.asarray(modalSimulator.eig_freq),
        source_weights=np.asarray(sourceWeights),
        zeta=np.asarray(zeta),
        freq_step=np.asarray(freqStep),
        modal_order=np.asarray(MODAL_ORDER),
        n_modes=np.asarray(N_MODES),
        target_freq=np.asarray(TARGET_FREQ),
    )

    return str(outputPath)


# =========================================================
# Main
# =========================================================

def main():
    experimentRooms = loadJson(ROOMS_JSON)
    summary = loadSummary()

    globalStart = time.perf_counter()

    for roomName, roomParams in experimentRooms.items():
        print()
        print("#" * 80)
        print(f"Room: {roomName}")
        print("#" * 80, flush=True)

        try:
            meshStart = time.perf_counter()

            meshPath, sourcePos = createRoomMesh(roomName, roomParams)

            meshTime = time.perf_counter() - meshStart

            modalStart = time.perf_counter()

            modalSimulator, sourceWeights = setupModalRoom(meshPath)

            modalTime = time.perf_counter() - modalStart

            summary["rooms"][roomName] = {
                "status": "ok",
                "mesh_path": str(meshPath),
                "source_position": list(sourcePos),
                "mesh_time_s": float(meshTime),
                "modal_setup_time_s": float(modalTime),
                "n_modes_converged": int(len(modalSimulator.eig_freq)),
                "min_mode_hz": float(np.min(modalSimulator.eig_freq)),
                "max_mode_hz": float(np.max(modalSimulator.eig_freq)),
            }

            saveJson(summary, SUMMARY_PATH)

        except Exception as e:
            summary["rooms"][roomName] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            saveJson(summary, SUMMARY_PATH)
            continue

        positionConfigs = roomParams["data"]["position_configs"]

        for configName, config in positionConfigs.items():
            for freqStep in FREQ_STEPS:

                for useImpedance in USE_IMPEDANCE_VALUES:
                    caseName = directCaseName(
                        roomName,
                        configName,
                        freqStep,
                        useImpedance,
                    )

                    try:
                        outputPath = runDirectCase(
                            meshPath=meshPath,
                            sourcePos=sourcePos,
                            roomName=roomName,
                            configName=configName,
                            config=config,
                            freqStep=freqStep,
                            useImpedance=useImpedance,
                        )

                        summary["cases"][caseName] = {
                            "status": "ok",
                            "type": "direct",
                            "room_name": roomName,
                            "config_name": configName,
                            "freq_step": freqStep,
                            "use_impedance": useImpedance,
                            "output_path": outputPath,
                        }

                    except Exception as e:
                        summary["cases"][caseName] = {
                            "status": "failed",
                            "type": "direct",
                            "room_name": roomName,
                            "config_name": configName,
                            "freq_step": freqStep,
                            "use_impedance": useImpedance,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }

                    saveJson(summary, SUMMARY_PATH)

                for zeta in ZETA_VALUES:
                    caseName = modalCaseName(
                        roomName,
                        configName,
                        freqStep,
                        zeta,
                    )

                    try:
                        outputPath = runModalCase(
                            modalSimulator=modalSimulator,
                            sourceWeights=sourceWeights,
                            sourcePos=sourcePos,
                            roomName=roomName,
                            configName=configName,
                            config=config,
                            freqStep=freqStep,
                            zeta=zeta,
                        )

                        summary["cases"][caseName] = {
                            "status": "ok",
                            "type": "modal",
                            "room_name": roomName,
                            "config_name": configName,
                            "freq_step": freqStep,
                            "zeta": zeta,
                            "output_path": outputPath,
                        }

                    except Exception as e:
                        summary["cases"][caseName] = {
                            "status": "failed",
                            "type": "modal",
                            "room_name": roomName,
                            "config_name": configName,
                            "freq_step": freqStep,
                            "zeta": zeta,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }

                    saveJson(summary, SUMMARY_PATH)

    totalTime = time.perf_counter() - globalStart

    summary["status"] = "finished"
    summary["total_runtime_s"] = float(totalTime)

    saveJson(summary, SUMMARY_PATH)

    print()
    print("=" * 80)
    print("Overnight sweep finished.")
    print(f"Results dir: {OUT_DIR}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Total runtime: {totalTime:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()