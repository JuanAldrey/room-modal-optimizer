from room_modal_optimizer.pipeline.pipeline import Pipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator

from itertools import product
from pathlib import Path
import numpy as np
import csv
import json
import time
import traceback

"""

Script used to carry out a brute force resolution of the gene space present in optimization_symetric.py.
Results between both methods were compared in order

"""


# =========================================================
# Configuración del experimento
# =========================================================

runLabel = "bruteforce_ga_symmetric_space"
outputDir = Path("data") / runLabel
outputDir.mkdir(parents=True, exist_ok=True)

resultsCsv = outputDir / "bruteforce_results.csv"
bestJson = outputDir / "best_result.json"

# Paso de discretización del espacio continuo.
# NO usar 0.01 salvo que sea un espacio muchísimo más chico.
vertexStep = 0.25
zStep = 0.2

# Límite de seguridad para no lanzar millones de simulaciones por error.
maxAllowedCases = 10000

minMicDistance = 0.5
nMics = 4

debugParams = True
debugFirstNCases = 10
saveParamsDebug = True
paramsDebugDir = outputDir / "params_debug"
paramsDebugDir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Espacio genético y sala base
# =========================================================

gene_space_config = {
    "vertices": {
        "V2": {"dx": [-0.50, 0.50], "dy": [-0.50, 0.50]},
        "V3": {"dx": [-0.50, 0.50], "dy": [-0.50, 0.50]},
    },
    "walls": {},
    "Z": {"low": 3.0, "high": 4.2},
}

base_params = {
    "data": {
        "vertices": {
            "V1": [-2.5, 0.0],
            "V2": [ 2.5, 0.0],
            "V3": [ 2.5, 4.0],
            "V4": [-2.5, 4.0],
        },
        "walls": {
            "W1": 0.0,
            "W2": 0.0,
            "W3": 0.0,
            "W4": 0.0,
        },
        "audience_area": {
            "V1": [-1.0, 1.0],
            "V2": [ 1.0, 1.0],
            "V3": [ 1.0, 2.5],
            "V4": [-1.0, 2.5],
        },
        "Z": 3.0,
        "source_pos": [[0.0, 3.0, 1.5]],
    }
}


# =========================================================
# Helpers
# =========================================================

def makeValues(low, high, step):
    values = np.arange(low, high + step / 2.0, step)
    return np.round(values, 6)


def valueLabel(value):
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def cloneBaseParams(baseParams):
    return json.loads(json.dumps(baseParams))


def makeSymmetricParams(baseParams, dxV2, dyV2, dxV3, dyV3, zValue):
    params = cloneBaseParams(baseParams)

    baseVertices = baseParams["data"]["vertices"]

    v2Base = np.asarray(baseVertices["V2"], dtype=float)
    v3Base = np.asarray(baseVertices["V3"], dtype=float)

    v2 = v2Base + np.asarray([dxV2, dyV2], dtype=float)
    v3 = v3Base + np.asarray([dxV3, dyV3], dtype=float)

    # keepSymmetry=True:
    # V1 es el espejo de V2 respecto del eje x=0.
    # V4 es el espejo de V3 respecto del eje x=0.
    v1 = np.asarray([-v2[0], v2[1]], dtype=float)
    v4 = np.asarray([-v3[0], v3[1]], dtype=float)

    params["data"]["vertices"] = {
        "V1": [float(v1[0]), float(v1[1])],
        "V2": [float(v2[0]), float(v2[1])],
        "V3": [float(v3[0]), float(v3[1])],
        "V4": [float(v4[0]), float(v4[1])],
    }

    params["data"]["Z"] = float(zValue)

    return params


def makeCaseKey(dxV2, dyV2, dxV3, dyV3, zValue):
    return (
        f"dxV2_{valueLabel(dxV2)}__"
        f"dyV2_{valueLabel(dyV2)}__"
        f"dxV3_{valueLabel(dxV3)}__"
        f"dyV3_{valueLabel(dyV3)}__"
        f"Z_{valueLabel(zValue)}"
    )


def appendRow(path, row):
    fileExists = path.exists()

    fieldnames = [
        "case_key",
        "room_name",
        "dxV2",
        "dyV2",
        "dxV3",
        "dyV3",
        "Z",
        "best_msfd",
        "best_mic_positions",
        "runtime_s",
        "status",
        "error",
    ]

    with open(path, "a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")

        if not fileExists:
            writer.writeheader()

        writer.writerow(row)


def loadCompletedCases(path):
    if not path.exists():
        return set()

    completed = set()

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            completed.add(row["case_key"])

    return completed


def readSuccessfulRows(path):
    if not path.exists():
        return []

    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            if row["status"] == "ok":
                row["best_msfd"] = float(row["best_msfd"])
                row["dxV2"] = float(row["dxV2"])
                row["dyV2"] = float(row["dyV2"])
                row["dxV3"] = float(row["dxV3"])
                row["dyV3"] = float(row["dyV3"])
                row["Z"] = float(row["Z"])
                rows.append(row)

    return rows


# =========================================================
# Main
# =========================================================

def main():
    dxV2Values = makeValues(
        gene_space_config["vertices"]["V2"]["dx"][0],
        gene_space_config["vertices"]["V2"]["dx"][1],
        vertexStep,
    )

    dyV2Values = makeValues(
        gene_space_config["vertices"]["V2"]["dy"][0],
        gene_space_config["vertices"]["V2"]["dy"][1],
        vertexStep,
    )

    dxV3Values = makeValues(
        gene_space_config["vertices"]["V3"]["dx"][0],
        gene_space_config["vertices"]["V3"]["dx"][1],
        vertexStep,
    )

    dyV3Values = makeValues(
        gene_space_config["vertices"]["V3"]["dy"][0],
        gene_space_config["vertices"]["V3"]["dy"][1],
        vertexStep,
    )

    zValues = makeValues(
        gene_space_config["Z"]["low"],
        gene_space_config["Z"]["high"],
        zStep,
    )

    totalCases = (
        len(dxV2Values)
        * len(dyV2Values)
        * len(dxV3Values)
        * len(dyV3Values)
        * len(zValues)
    )

    print("=" * 80)
    print("Brute force sobre espacio genético simétrico")
    print("=" * 80)
    print(f"vertexStep: {vertexStep}")
    print(f"zStep: {zStep}")
    print(f"Total cases: {totalCases}")
    print(f"Results CSV: {resultsCsv}")
    print("=" * 80)

    if totalCases > maxAllowedCases:
        raise ValueError(
            f"El espacio tiene {totalCases} casos, supera maxAllowedCases={maxAllowedCases}. "
            "Aumentá vertexStep/zStep o subí maxAllowedCases si estás seguro."
        )

    mesher = Mesher()
    directSimulator = DirectSimulator()
    evaluator = Evaluator()

    pipeline = Pipeline(
        mesher=mesher,
        directSimulator=directSimulator,
        evaluator=evaluator,
    )

    # No modifica la clase Pipeline: sólo apaga plots para que la fuerza bruta no genere miles de imágenes.
    pipeline.savePlantPlot = False
    pipeline.saveMicPlots = False

    completedCases = loadCompletedCases(resultsCsv)

    globalStart = time.perf_counter()
    caseIndex = 0

    for dxV2, dyV2, dxV3, dyV3, zValue in product(
        dxV2Values,
        dyV2Values,
        dxV3Values,
        dyV3Values,
        zValues,
    ):
        caseIndex += 1

        caseKey = makeCaseKey(dxV2, dyV2, dxV3, dyV3, zValue)
        roomName = f"{runLabel}_{caseKey}"

        if caseKey in completedCases:
            print(f"[{caseIndex}/{totalCases}] Skipping completed: {caseKey}")
            continue

        print()
        print("#" * 80)
        print(f"[{caseIndex}/{totalCases}] Running {caseKey}")
        print("#" * 80)

        params = makeSymmetricParams(
            baseParams=base_params,
            dxV2=dxV2,
            dyV2=dyV2,
            dxV3=dxV3,
            dyV3=dyV3,
            zValue=zValue,
        )

        print("Generated params:")
        print(json.dumps(params, indent=4, ensure_ascii=False))

        caseStart = time.perf_counter()

        try:
            result = pipeline.run(
                params=params,
                room_name=roomName,
                minMicDistance=minMicDistance,
                nMics=nMics,
            )

            runtime = time.perf_counter() - caseStart

            if result is None:
                row = {
                    "case_key": caseKey,
                    "room_name": roomName,
                    "dxV2": dxV2,
                    "dyV2": dyV2,
                    "dxV3": dxV3,
                    "dyV3": dyV3,
                    "Z": zValue,
                    "best_msfd": "",
                    "best_mic_positions": "",
                    "runtime_s": runtime,
                    "status": "mesh_failed",
                    "error": "",
                }
            else:
                bestMsfd, bestMicPositions = result

                row = {
                    "case_key": caseKey,
                    "room_name": roomName,
                    "dxV2": dxV2,
                    "dyV2": dyV2,
                    "dxV3": dxV3,
                    "dyV3": dyV3,
                    "Z": zValue,
                    "best_msfd": float(bestMsfd),
                    "best_mic_positions": json.dumps(np.asarray(bestMicPositions).tolist()),
                    "runtime_s": runtime,
                    "status": "ok",
                    "error": "",
                }

                print(f"Best MSFD: {bestMsfd:.4f}")
                print("Best mic positions:")
                print(bestMicPositions)

        except Exception:
            runtime = time.perf_counter() - caseStart

            row = {
                "case_key": caseKey,
                "room_name": roomName,
                "dxV2": dxV2,
                "dyV2": dyV2,
                "dxV3": dxV3,
                "dyV3": dyV3,
                "Z": zValue,
                "best_msfd": "",
                "best_mic_positions": "",
                "runtime_s": runtime,
                "status": "error",
                "error": traceback.format_exc(),
            }

            print("ERROR:")
            print(row["error"])

        appendRow(resultsCsv, row)

    successfulRows = readSuccessfulRows(resultsCsv)

    if len(successfulRows) == 0:
        print("No hubo casos exitosos.")
        return

    successfulRows = sorted(successfulRows, key=lambda row: row["best_msfd"])
    bestRow = successfulRows[0]

    bestParams = makeSymmetricParams(
        baseParams=base_params,
        dxV2=bestRow["dxV2"],
        dyV2=bestRow["dyV2"],
        dxV3=bestRow["dxV3"],
        dyV3=bestRow["dyV3"],
        zValue=bestRow["Z"],
    )

    with open(bestJson, "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_row": bestRow,
                "best_params": bestParams,
                "search_space": {
                    "vertex_step": vertexStep,
                    "z_step": zStep,
                    "dxV2_values": dxV2Values.tolist(),
                    "dyV2_values": dyV2Values.tolist(),
                    "dxV3_values": dxV3Values.tolist(),
                    "dyV3_values": dyV3Values.tolist(),
                    "z_values": zValues.tolist(),
                    "total_cases": totalCases,
                },
                "base_params": base_params,
                "gene_space_config": gene_space_config,
                "pipeline_config": {
                    "minMicDistance": minMicDistance,
                    "nMics": nMics,
                },
            },
            file,
            indent=4,
            ensure_ascii=False,
        )

    totalRuntime = time.perf_counter() - globalStart

    print()
    print("=" * 80)
    print("Finished brute force.")
    print("=" * 80)
    print(f"Total runtime: {totalRuntime:.2f} s")
    print(f"Best MSFD: {bestRow['best_msfd']:.4f}")
    print(f"Best case: {bestRow['case_key']}")
    print(f"Best JSON: {bestJson}")

    print()
    print("Top 10:")
    for index, row in enumerate(successfulRows[:10], start=1):
        print(
            f"{index:02d}. MSFD={row['best_msfd']:.4f} | "
            f"dxV2={row['dxV2']:.3f}, dyV2={row['dyV2']:.3f}, "
            f"dxV3={row['dxV3']:.3f}, dyV3={row['dyV3']:.3f}, "
            f"Z={row['Z']:.3f}"
        )


if __name__ == "__main__":
    main()