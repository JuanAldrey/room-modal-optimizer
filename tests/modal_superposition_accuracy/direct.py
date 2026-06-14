import json
import time
import traceback
from pathlib import Path

import numpy as np

from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator

# =========================================================
# Paths
# =========================================================

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

ROOMS_JSON = RESULTS_DIR / "single_controlled.json"

# =========================================================
# Config
# =========================================================

DIRECT_ORDER = 1
RUN_LABEL = f"single_direct_{DIRECT_ORDER}"

DIRECT_RESULTS_DIR = RESULTS_DIR / RUN_LABEL
DIRECT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = DIRECT_RESULTS_DIR / f"{RUN_LABEL}_summary.json"

EXPORT_DIRECT = False

# Si mañana querés cambiar el rango, revisamos DirectSimulator.
# Por ahora usa el rango interno/default de tu DirectSimulator.
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
        "n_total": 0,
        "n_done": 0,
        "n_failed": 0,
        "cases": {}
    }


def cleanRoomParams(roomParams):
    """
    El Mesher solo necesita vertices, walls y Z.
    Sacamos position_configs para no pasarle campos extra.
    """
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


def getCaseName(roomName, configName):
    return f"{roomName}_{configName}_direct"


def getCaseOutputPath(caseName):
    return DIRECT_RESULTS_DIR / f"{caseName}.npz"


def runDirectCase(roomName, roomParams, configName, config):
    caseName = getCaseName(roomName, configName)

    print()
    print("=" * 80)
    print(f"Running case: {caseName}")
    print("=" * 80, flush=True)

    sourcePos = tuple(config["source"])
    micPositions = sortedMicPositions(config)

    cleanParams = cleanRoomParams(roomParams)

    caseStart = time.perf_counter()

    # -----------------------------------------------------
    # Mesh
    # -----------------------------------------------------
    meshStart = time.perf_counter()

    mesher = Mesher()

    meshPath = mesher.create(
        cleanParams,
        room_name=caseName,
        visualize=False,
        source_pos=sourcePos,
    )

    meshTime = time.perf_counter() - meshStart

    print(f"Mesh ready: {meshPath}")
    print(f"Mesh time: {meshTime:.2f} s", flush=True)

    # -----------------------------------------------------
    # Direct simulation
    # -----------------------------------------------------
    simStart = time.perf_counter()

    directSimulator = DirectSimulator()

    freqs, splResponses = directSimulator.simulate(
        meshPath,
        mic_positions=micPositions,
        room_name=caseName,
        export=EXPORT_DIRECT,
    )

    simTime = time.perf_counter() - simStart
    totalTime = time.perf_counter() - caseStart

    print(f"Simulation time: {simTime:.2f} s")
    print(f"Total case time: {totalTime:.2f} s", flush=True)

    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------
    outputPath = getCaseOutputPath(caseName)

    np.savez_compressed(
        outputPath,
        freqs=np.asarray(freqs),
        spl_responses=np.asarray(splResponses),
        source_position=np.asarray(sourcePos),
        mic_positions=np.asarray(micPositions),
    )

    caseResult = {
        "status": "ok",
        "room_name": roomName,
        "config_name": configName,
        "case_name": caseName,
        "output_path": str(outputPath),
        "source_position": list(sourcePos),
        "mic_positions": [list(mic) for mic in micPositions],
        "n_freqs": int(len(freqs)),
        "n_mics": int(np.asarray(splResponses).shape[1]),
        "mesh_time_s": float(meshTime),
        "simulation_time_s": float(simTime),
        "total_time_s": float(totalTime),
    }

    return caseResult


# =========================================================
# Main
# =========================================================

def main():
    experimentRooms = loadJson(ROOMS_JSON)
    summary = loadSummary()

    allCases = []

    for roomName, roomParams in experimentRooms.items():
        positionConfigs = roomParams["data"]["position_configs"]

        for configName, config in positionConfigs.items():
            caseName = getCaseName(roomName, configName)
            allCases.append((caseName, roomName, roomParams, configName, config))

    summary["n_total"] = len(allCases)
    saveJson(summary, SUMMARY_PATH)

    print(f"Total direct cases: {len(allCases)}")
    print(f"Results dir: {DIRECT_RESULTS_DIR}")
    print(f"Summary path: {SUMMARY_PATH}")
    print()

    globalStart = time.perf_counter()

    for caseIndex, (caseName, roomName, roomParams, configName, config) in enumerate(allCases, start=1):
        outputPath = getCaseOutputPath(caseName)

        alreadyDone = (
            caseName in summary["cases"]
            and summary["cases"][caseName].get("status") == "ok"
            and outputPath.exists()
        )

        if SKIP_COMPLETED and alreadyDone:
            print(f"[{caseIndex}/{len(allCases)}] Skipping completed case: {caseName}")
            continue

        print(f"[{caseIndex}/{len(allCases)}] Starting {caseName}", flush=True)

        try:
            caseResult = runDirectCase(
                roomName=roomName,
                roomParams=roomParams,
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

        saveJson(summary, SUMMARY_PATH)

        print(f"Progress: ok={nDone}, failed={nFailed}, total={len(allCases)}", flush=True)

    totalTime = time.perf_counter() - globalStart

    summary["status"] = "finished"
    summary["total_runtime_s"] = float(totalTime)

    saveJson(summary, SUMMARY_PATH)

    print()
    print("=" * 80)
    print("Direct simulation batch finished.")
    print(f"OK: {summary['n_done']}")
    print(f"Failed: {summary['n_failed']}")
    print(f"Total runtime: {totalTime:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()