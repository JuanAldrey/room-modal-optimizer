import json
import time
import shutil
import traceback
from pathlib import Path

import numpy as np

from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.modal_simulator import ModalSimulator


# =========================================================
# Paths
# =========================================================

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

ROOMS_JSON = RESULTS_DIR / "single_controlled.json"


# =========================================================
# Config
# =========================================================

# Usar el mismo orden que en DirectSimulator.
# Si DirectSimulator usa Lagrange 1, poné 1.
# Si DirectSimulator usa Lagrange 2, poné 2.
MODAL_ORDER = 2

N_MODES = 300
TARGET_FREQ = 100.0
MODAL_TOL = 1e-8

# Para comparar contra directo rígido sin damping, usar 0.
ZETA = 0.005

SOURCE_STRENGTH = 0.01

FREQS = np.arange(20.0, 101.0, 2.0)

RUN_LABEL = (
    f"single_modal"
    f"_ZETA_{ZETA}"
)

MODAL_RESULTS_DIR = RESULTS_DIR / RUN_LABEL
MODAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = MODAL_RESULTS_DIR / f"{RUN_LABEL}_summary.json"

SKIP_COMPLETED = True


# =========================================================
# Helpers
# =========================================================

def checkDependencies():
    if shutil.which("gcc") is None:
        raise RuntimeError(
            "No se encontró gcc. Instalalo con: sudo apt install build-essential"
        )


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
        "source_model": "point_source",
        "mesh_strategy": "one_mesh_per_room_no_source",
        "modal_order": MODAL_ORDER,
        "n_modes_requested": N_MODES,
        "target_freq": TARGET_FREQ,
        "modal_tol": MODAL_TOL,
        "zeta": ZETA,
        "source_strength": SOURCE_STRENGTH,
        "freq_min": float(FREQS[0]),
        "freq_max": float(FREQS[-1]),
        "freq_step": float(FREQS[1] - FREQS[0]),
        "n_total": 0,
        "n_done": 0,
        "n_failed": 0,
        "rooms": {},
        "cases": {},
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


def getRoomModalName(roomName):
    return f"{roomName}_{RUN_LABEL}"


def getCaseName(roomName, configName):
    return f"{roomName}_{configName}_{RUN_LABEL}"


def getCaseOutputPath(caseName):
    return MODAL_RESULTS_DIR / f"{caseName}.npz"


def getRoomOutputPath(roomName):
    return MODAL_RESULTS_DIR / f"{getRoomModalName(roomName)}_room_modal.npz"


def sortModes(modalSimulator):
    pairs = sorted(
        zip(modalSimulator.eig_freq, modalSimulator.eig_vector),
        key=lambda x: x[0]
    )

    modalSimulator.eig_freq = [freq for freq, vec in pairs]
    modalSimulator.eig_vector = [vec for freq, vec in pairs]


def computeSplFromTransfer(H):
    return 20.0 * np.log10(np.abs(H) + 1e-12)


def updateSummaryCounts(summary):
    nDone = sum(
        1 for case in summary["cases"].values()
        if case.get("status") == "ok"
    )

    nFailed = sum(
        1 for case in summary["cases"].values()
        if case.get("status") == "failed"
    )

    summary["n_done"] = nDone
    summary["n_failed"] = nFailed


# =========================================================
# Modal por room
# =========================================================

def runModalRoom(roomName, roomParams):
    modalRoomName = getRoomModalName(roomName)

    print()
    print("=" * 80)
    print(f"Running modal room: {modalRoomName}")
    print("=" * 80, flush=True)

    cleanParams = cleanRoomParams(roomParams)

    roomStart = time.perf_counter()

    # -----------------------------------------------------
    # Mesh SIN fuente
    # -----------------------------------------------------

    meshStart = time.perf_counter()

    mesher = Mesher()

    meshPath = mesher.create(
        cleanParams,
        room_name=modalRoomName,
        visualize=False,
    )

    meshTime = time.perf_counter() - meshStart

    print(f"Mesh ready: {meshPath}")
    print(f"Mesh time: {meshTime:.2f} s", flush=True)

    # -----------------------------------------------------
    # Modal simulation
    # -----------------------------------------------------

    modalStart = time.perf_counter()

    modalSimulator = ModalSimulator()
    modalSimulator.room_name = modalRoomName

    modalSimulator.loadMesh(meshPath)
    modalSimulator.setup(order=MODAL_ORDER)

    modalSimulator.computeModalAnalysis(
        target_freq=TARGET_FREQ,
        n_modes=N_MODES,
        tol=MODAL_TOL,
    )

    modalSimulator.obtainModes()
    sortModes(modalSimulator)

    modalTime = time.perf_counter() - modalStart
    totalRoomTime = time.perf_counter() - roomStart

    eigFreq = np.asarray(modalSimulator.eig_freq, dtype=float)

    if len(eigFreq) == 0:
        raise RuntimeError("No convergió ningún modo.")

    print(f"Converged modes: {len(eigFreq)}")
    print(f"Min mode: {np.min(eigFreq):.6f} Hz")
    print(f"Max mode: {np.max(eigFreq):.6f} Hz")
    print(f"First modes: {eigFreq[:10]}")
    print(f"Last modes: {eigFreq[-10:]}")
    print(f"Modal time: {modalTime:.2f} s")
    print(f"Total room time: {totalRoomTime:.2f} s", flush=True)

    np.savez_compressed(
        getRoomOutputPath(roomName),
        eig_freq=eigFreq,
        mesh_path=np.asarray(str(meshPath)),
        modal_order=np.asarray(MODAL_ORDER),
        n_modes_requested=np.asarray(N_MODES),
        n_modes_converged=np.asarray(len(eigFreq)),
        target_freq=np.asarray(TARGET_FREQ),
        zeta=np.asarray(ZETA),
        source_model=np.asarray("point_source"),
        mesh_strategy=np.asarray("one_mesh_per_room_no_source"),
    )

    roomResult = {
        "status": "ok",
        "room_name": roomName,
        "modal_room_name": modalRoomName,
        "mesh_path": str(meshPath),
        "room_output_path": str(getRoomOutputPath(roomName)),
        "modal_order": int(MODAL_ORDER),
        "n_modes_requested": int(N_MODES),
        "n_modes_converged": int(len(eigFreq)),
        "min_mode_hz": float(np.min(eigFreq)),
        "max_mode_hz": float(np.max(eigFreq)),
        "target_freq_hz": float(TARGET_FREQ),
        "zeta": float(ZETA),
        "source_model": "point_source",
        "mesh_strategy": "one_mesh_per_room_no_source",
        "mesh_time_s": float(meshTime),
        "modal_time_s": float(modalTime),
        "total_room_time_s": float(totalRoomTime),
    }

    return modalSimulator, roomResult


# =========================================================
# Modal por config
# =========================================================

def runModalCase(modalSimulator, roomName, configName, config):
    caseName = getCaseName(roomName, configName)

    print(f"Running modal case: {caseName}", flush=True)

    sourcePos = tuple(config["source"])
    micPositions = sortedMicPositions(config)

    caseStart = time.perf_counter()

    # -----------------------------------------------------
    # Cache de puntos
    # -----------------------------------------------------

    cacheStart = time.perf_counter()

    cache = modalSimulator.buildPointCache(
        sourcePositions=[sourcePos],
        receiverPositions=micPositions,
    )

    cacheTime = time.perf_counter() - cacheStart

    # -----------------------------------------------------
    # Transferencia modal rápida
    # -----------------------------------------------------

    transferStart = time.perf_counter()

    H = modalSimulator.modalTransferFromCache(
        cache=cache,
        sourceIndex=0,
        freqs=FREQS,
        zeta=ZETA,
        sourceStrength=SOURCE_STRENGTH,
    )

    transferTime = time.perf_counter() - transferStart

    splResponses = computeSplFromTransfer(H)

    totalCaseTime = time.perf_counter() - caseStart

    outputPath = getCaseOutputPath(caseName)

    np.savez_compressed(
        outputPath,
        freqs=np.asarray(FREQS),
        H_modal=np.asarray(H),
        spl_responses=np.asarray(splResponses),
        source_position=np.asarray(sourcePos),
        mic_positions=np.asarray(micPositions),
        eig_freq=np.asarray(modalSimulator.eig_freq),
        modal_order=np.asarray(MODAL_ORDER),
        n_modes_requested=np.asarray(N_MODES),
        n_modes_converged=np.asarray(len(modalSimulator.eig_freq)),
        target_freq=np.asarray(TARGET_FREQ),
        zeta=np.asarray(ZETA),
        source_strength=np.asarray(SOURCE_STRENGTH),
        source_model=np.asarray("point_source"),
        mesh_strategy=np.asarray("one_mesh_per_room_no_source"),
    )

    caseResult = {
        "status": "ok",
        "room_name": roomName,
        "config_name": configName,
        "case_name": caseName,
        "output_path": str(outputPath),
        "source_position": list(sourcePos),
        "mic_positions": [list(mic) for mic in micPositions],
        "n_freqs": int(len(FREQS)),
        "n_mics": int(np.asarray(splResponses).shape[0]),
        "modal_order": int(MODAL_ORDER),
        "n_modes_requested": int(N_MODES),
        "n_modes_converged": int(len(modalSimulator.eig_freq)),
        "target_freq_hz": float(TARGET_FREQ),
        "zeta": float(ZETA),
        "source_strength": float(SOURCE_STRENGTH),
        "source_model": "point_source",
        "mesh_strategy": "one_mesh_per_room_no_source",
        "cache_time_s": float(cacheTime),
        "transfer_time_s": float(transferTime),
        "total_case_time_s": float(totalCaseTime),
    }

    return caseResult


# =========================================================
# Main
# =========================================================

def main():
    checkDependencies()

    experimentRooms = loadJson(ROOMS_JSON)
    summary = loadSummary()

    allCases = []

    for roomName, roomParams in experimentRooms.items():
        positionConfigs = roomParams["data"]["position_configs"]

        for configName, config in positionConfigs.items():
            caseName = getCaseName(roomName, configName)
            allCases.append((caseName, roomName, configName, config))

    summary["n_total"] = len(allCases)
    saveJson(summary, SUMMARY_PATH)

    print(f"Total modal cases: {len(allCases)}")
    print(f"Rooms: {len(experimentRooms)}")
    print(f"Results dir: {MODAL_RESULTS_DIR}")
    print(f"Summary path: {SUMMARY_PATH}")
    print()
    print("Config:")
    print(f"  MODAL_ORDER: {MODAL_ORDER}")
    print(f"  N_MODES: {N_MODES}")
    print(f"  TARGET_FREQ: {TARGET_FREQ}")
    print(f"  ZETA: {ZETA}")
    print(f"  SOURCE_STRENGTH: {SOURCE_STRENGTH}")
    print(f"  FREQS: {FREQS[0]}-{FREQS[-1]} Hz, step {FREQS[1] - FREQS[0]} Hz")
    print(f"  Source model: point_source")
    print(f"  Mesh strategy: one_mesh_per_room_no_source")
    print()

    globalStart = time.perf_counter()

    for roomIndex, (roomName, roomParams) in enumerate(experimentRooms.items(), start=1):
        print()
        print("#" * 80)
        print(f"Room [{roomIndex}/{len(experimentRooms)}]: {roomName}")
        print("#" * 80, flush=True)

        positionConfigs = roomParams["data"]["position_configs"]

        roomCaseNames = [
            getCaseName(roomName, configName)
            for configName in positionConfigs.keys()
        ]

        roomAlreadyDone = all(
            caseName in summary["cases"]
            and summary["cases"][caseName].get("status") == "ok"
            and getCaseOutputPath(caseName).exists()
            for caseName in roomCaseNames
        )

        if SKIP_COMPLETED and roomAlreadyDone:
            print(f"Skipping completed room: {roomName}")
            continue

        try:
            modalSimulator, roomResult = runModalRoom(
                roomName=roomName,
                roomParams=roomParams,
            )

            summary["rooms"][roomName] = roomResult
            saveJson(summary, SUMMARY_PATH)

        except Exception as e:
            print(f"ERROR in modal room {roomName}: {e}", flush=True)

            summary["rooms"][roomName] = {
                "status": "failed",
                "room_name": roomName,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

            for configName in positionConfigs.keys():
                caseName = getCaseName(roomName, configName)
                summary["cases"][caseName] = {
                    "status": "failed",
                    "room_name": roomName,
                    "config_name": configName,
                    "case_name": caseName,
                    "error": f"Modal room failed: {e}",
                }

            updateSummaryCounts(summary)
            saveJson(summary, SUMMARY_PATH)
            continue

        for configName, config in positionConfigs.items():
            caseName = getCaseName(roomName, configName)
            outputPath = getCaseOutputPath(caseName)

            alreadyDone = (
                caseName in summary["cases"]
                and summary["cases"][caseName].get("status") == "ok"
                and outputPath.exists()
            )

            if SKIP_COMPLETED and alreadyDone:
                print(f"Skipping completed case: {caseName}")
                continue

            try:
                caseResult = runModalCase(
                    modalSimulator=modalSimulator,
                    roomName=roomName,
                    configName=configName,
                    config=config,
                )

                summary["cases"][caseName] = caseResult

            except Exception as e:
                print(f"ERROR in case {caseName}: {e}", flush=True)

                summary["cases"][caseName] = {
                    "status": "failed",
                    "room_name": roomName,
                    "config_name": configName,
                    "case_name": caseName,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

            updateSummaryCounts(summary)
            saveJson(summary, SUMMARY_PATH)

            print(
                f"Progress: ok={summary['n_done']}, "
                f"failed={summary['n_failed']}, "
                f"total={len(allCases)}",
                flush=True,
            )

    totalTime = time.perf_counter() - globalStart

    summary["status"] = "finished"
    summary["total_runtime_s"] = float(totalTime)

    updateSummaryCounts(summary)
    saveJson(summary, SUMMARY_PATH)

    print()
    print("=" * 80)
    print("Modal simulation batch finished.")
    print(f"OK: {summary['n_done']}")
    print(f"Failed: {summary['n_failed']}")
    print(f"Total runtime: {totalTime:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()