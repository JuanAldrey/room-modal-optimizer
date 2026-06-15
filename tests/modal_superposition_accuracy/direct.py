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

ROOMS_JSON = RESULTS_DIR / "single_square_fixed_source_many_mics.json"


# =========================================================
# Config
# =========================================================

DIRECT_ORDER = 1

RUN_LABEL = f"direct_{DIRECT_ORDER}"

DIRECT_RESULTS_DIR = RESULTS_DIR / RUN_LABEL
DIRECT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = DIRECT_RESULTS_DIR / f"{RUN_LABEL}_summary.json"

EXPORT_DIRECT = False

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
        "source_model": "fixed_surface_source",
        "mesh_strategy": "one_mesh_per_room_with_fixed_source",
        "direct_order": DIRECT_ORDER,
        "n_total": 0,
        "n_done": 0,
        "n_failed": 0,
        "rooms": {},
        "cases": {},
    }


def cleanRoomParams(roomParams):
    """
    El Mesher solo necesita vertices, walls y Z.
    Sacamos source y position_configs para no pasarle campos extra.
    """
    data = roomParams["data"]

    return {
        "data": {
            "vertices": data["vertices"],
            "walls": data["walls"],
            "Z": data["Z"],
        }
    }


def getRoomSource(roomParams):
    """
    Fuente fija a nivel de sala.

    Formato nuevo esperado:
        roomParams["data"]["source"]

    Fallback:
        si no existe, intenta tomar la fuente del primer config,
        pero verifica que todas las configs tengan la misma.
    """
    data = roomParams["data"]

    if "source" in data:
        return tuple(data["source"])

    positionConfigs = data["position_configs"]

    firstConfig = next(iter(positionConfigs.values()))

    if "source" not in firstConfig:
        raise KeyError(
            "No encontré fuente fija. Agregá data['source'] "
            "o source dentro de cada config."
        )

    sourcePos = tuple(firstConfig["source"])

    for configName, config in positionConfigs.items():
        if tuple(config.get("source", sourcePos)) != sourcePos:
            raise ValueError(
                "Este script asume fuente fija. "
                f"Pero la config {configName} tiene otra source."
            )

    return sourcePos


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


def getRoomDirectName(roomName):
    return f"{roomName}_direct_fixed_source"


def getCaseName(roomName, configName):
    return f"{roomName}_{configName}_direct"


def getRoomOutputPath(roomName):
    return DIRECT_RESULTS_DIR / f"{getRoomDirectName(roomName)}.npz"


def getCaseOutputPath(caseName):
    return DIRECT_RESULTS_DIR / f"{caseName}.npz"


def getNumMicsFromSpl(splResponses, micPositions, freqs):
    splArray = np.asarray(splResponses)

    if splArray.ndim != 2:
        return len(micPositions)

    if splArray.shape[0] == len(micPositions):
        return int(splArray.shape[0])

    if splArray.shape[1] == len(micPositions):
        return int(splArray.shape[1])

    if splArray.shape[0] == len(freqs):
        return int(splArray.shape[1])

    return int(splArray.shape[0])


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
# Direct por room
# =========================================================

def runDirectRoom(roomName, roomParams):
    directRoomName = getRoomDirectName(roomName)

    print()
    print("=" * 80)
    print(f"Running direct room mesh: {directRoomName}")
    print("=" * 80, flush=True)

    cleanParams = cleanRoomParams(roomParams)
    sourcePos = getRoomSource(roomParams)

    roomStart = time.perf_counter()

    # -----------------------------------------------------
    # Mesh con fuente fija
    # -----------------------------------------------------

    meshStart = time.perf_counter()

    mesher = Mesher()

    meshPath = mesher.create(
        cleanParams,
        room_name=directRoomName,
        visualize=False,
        source_pos=sourcePos,
    )

    meshTime = time.perf_counter() - meshStart
    totalRoomTime = time.perf_counter() - roomStart

    print(f"Mesh ready: {meshPath}")
    print(f"Mesh time: {meshTime:.2f} s")
    print(f"Fixed source: {sourcePos}", flush=True)

    roomOutputPath = getRoomOutputPath(roomName)

    np.savez_compressed(
        roomOutputPath,
        mesh_path=np.asarray(str(meshPath)),
        source_position=np.asarray(sourcePos),
        source_model=np.asarray("fixed_surface_source"),
        mesh_strategy=np.asarray("one_mesh_per_room_with_fixed_source"),
        direct_order=np.asarray(DIRECT_ORDER),
    )

    roomResult = {
        "status": "ok",
        "room_name": roomName,
        "direct_room_name": directRoomName,
        "mesh_path": str(meshPath),
        "room_output_path": str(roomOutputPath),
        "source_position": list(sourcePos),
        "source_model": "fixed_surface_source",
        "mesh_strategy": "one_mesh_per_room_with_fixed_source",
        "direct_order": int(DIRECT_ORDER),
        "mesh_time_s": float(meshTime),
        "total_room_time_s": float(totalRoomTime),
    }

    return meshPath, sourcePos, roomResult


# =========================================================
# Direct por config de mics
# =========================================================

def runDirectCase(
    roomName,
    configName,
    config,
    meshPath,
    sourcePos,
):
    caseName = getCaseName(roomName, configName)

    print()
    print("=" * 80)
    print(f"Running case: {caseName}")
    print("=" * 80, flush=True)

    micPositions = sortedMicPositions(config)

    caseStart = time.perf_counter()

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
        mesh_path=np.asarray(str(meshPath)),
        source_model=np.asarray("fixed_surface_source"),
        mesh_strategy=np.asarray("one_mesh_per_room_with_fixed_source"),
        direct_order=np.asarray(DIRECT_ORDER),
    )

    nMics = getNumMicsFromSpl(
        splResponses=splResponses,
        micPositions=micPositions,
        freqs=freqs,
    )

    caseResult = {
        "status": "ok",
        "room_name": roomName,
        "config_name": configName,
        "case_name": caseName,
        "output_path": str(outputPath),
        "mesh_path": str(meshPath),
        "source_position": list(sourcePos),
        "mic_positions": [list(mic) for mic in micPositions],
        "n_freqs": int(len(freqs)),
        "n_mics": int(nMics),
        "source_model": "fixed_surface_source",
        "mesh_strategy": "one_mesh_per_room_with_fixed_source",
        "direct_order": int(DIRECT_ORDER),
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
            allCases.append((caseName, roomName, configName, config))

    summary["n_total"] = len(allCases)
    saveJson(summary, SUMMARY_PATH)

    print(f"Total direct cases: {len(allCases)}")
    print(f"Rooms: {len(experimentRooms)}")
    print(f"Results dir: {DIRECT_RESULTS_DIR}")
    print(f"Summary path: {SUMMARY_PATH}")
    print()
    print("Config:")
    print(f"  DIRECT_ORDER: {DIRECT_ORDER}")
    print(f"  Source model: fixed_surface_source")
    print(f"  Mesh strategy: one_mesh_per_room_with_fixed_source")
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
            meshPath, sourcePos, roomResult = runDirectRoom(
                roomName=roomName,
                roomParams=roomParams,
            )

            summary["rooms"][roomName] = roomResult
            saveJson(summary, SUMMARY_PATH)

        except Exception as e:
            print(f"ERROR in direct room {roomName}: {e}", flush=True)

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
                    "error": f"Direct room failed: {e}",
                }

            updateSummaryCounts(summary)
            saveJson(summary, SUMMARY_PATH)
            continue

        for configIndex, (configName, config) in enumerate(positionConfigs.items(), start=1):
            caseName = getCaseName(roomName, configName)
            outputPath = getCaseOutputPath(caseName)

            alreadyDone = (
                caseName in summary["cases"]
                and summary["cases"][caseName].get("status") == "ok"
                and outputPath.exists()
            )

            if SKIP_COMPLETED and alreadyDone:
                print(
                    f"[{configIndex}/{len(positionConfigs)}] "
                    f"Skipping completed case: {caseName}"
                )
                continue

            print(
                f"[{configIndex}/{len(positionConfigs)}] "
                f"Starting {caseName}",
                flush=True,
            )

            try:
                caseResult = runDirectCase(
                    roomName=roomName,
                    configName=configName,
                    config=config,
                    meshPath=meshPath,
                    sourcePos=sourcePos,
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
    print("Direct simulation batch finished.")
    print(f"OK: {summary['n_done']}")
    print(f"Failed: {summary['n_failed']}")
    print(f"Total runtime: {totalTime:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()