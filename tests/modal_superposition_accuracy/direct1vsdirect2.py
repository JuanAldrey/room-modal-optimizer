import csv
import json
import time
from pathlib import Path

import numpy as np

from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator


# =========================================================
# Config rápida
# =========================================================

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

RUN_LABEL = "direct_many_mics_order_sweep_multiroom"
OUT_DIR = RESULTS_DIR / RUN_LABEL
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESPONSES_DIR = OUT_DIR / "responses"
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / "summary.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"
COMBOS_CSV = OUT_DIR / "combo_msfd.csv"

FREQS = np.arange(20.0, 201.0, 2.0)

DIRECT_ORDERS = [1, 2]

IMPEDANCE_VALUES = [
    2.0 + 0j,
    20.0 + 0j,
    40.0 + 0j,
    200.0 + 0j,
    400.0 + 0j,
]

N_CANDIDATE_MICS = 200
N_MICS_PER_COMBO = 4
N_COMBOS = 100

MIC_HEIGHT = 1.2
AUDIENCE_MARGIN = 0.05
MIN_COMBO_DISTANCE = 0.25

RANDOM_SEED = 1234

SKIP_COMPLETED = True


# =========================================================
# Rooms de prueba
# =========================================================

ROOMS = {
    "room_rect_3x5": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [0.0, 5.0],
                "V3": [3.0, 5.0],
                "V4": [3.0, 0.0],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
            },
            "audience_area": {
                "V1": [1.0, 0.2],
                "V2": [1.0, 2.0],
                "V3": [2.0, 2.0],
                "V4": [2.0, 0.2],
            },
            "Z": 3.0,
            "source_pos": [1.5, 4.0, 1.5],
        }
    },

    "room_trapezoid_4w": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [0.3, 5.0],
                "V3": [3.4, 4.7],
                "V4": [3.0, 0.2],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
            },
            "audience_area": {
                "V1": [1.0, 0.4],
                "V2": [1.1, 2.3],
                "V3": [2.3, 2.2],
                "V4": [2.2, 0.4],
            },
            "Z": 3.0,
            "source_pos": [1.6, 4.0, 1.5],
        }
    },

    "room_pentagon_5w": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [0.0, 5.0],
                "V3": [2.0, 5.4],
                "V4": [3.5, 4.2],
                "V5": [3.2, 0.1],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
                "W5": 0.0,
            },
            "audience_area": {
                "V1": [1.0, 0.5],
                "V2": [1.0, 2.3],
                "V3": [2.2, 2.3],
                "V4": [2.2, 0.5],
            },
            "Z": 3.1,
            "source_pos": [1.7, 4.2, 1.5],
        }
    },

    "room_hex_6w": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.2],
                "V2": [0.2, 5.0],
                "V3": [1.4, 5.5],
                "V4": [3.2, 5.0],
                "V5": [3.6, 0.4],
                "V6": [2.0, 0.0],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
                "W5": 0.0,
                "W6": 0.0,
            },
            "audience_area": {
                "V1": [1.0, 0.6],
                "V2": [1.0, 2.5],
                "V3": [2.4, 2.5],
                "V4": [2.4, 0.6],
            },
            "Z": 3.0,
            "source_pos": [1.8, 4.2, 1.5],
        }
    },

    "room_skew_4w": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [-0.2, 5.2],
                "V3": [3.1, 5.0],
                "V4": [3.4, -0.1],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
            },
            "audience_area": {
                "V1": [1.0, 0.5],
                "V2": [1.0, 2.4],
                "V3": [2.3, 2.4],
                "V4": [2.3, 0.5],
            },
            "Z": 2.8,
            "source_pos": [1.5, 4.1, 1.4],
        }
    },
}


# =========================================================
# Helpers
# =========================================================

def cleanRoomParams(roomParams):
    data = roomParams["data"]

    return {
        "data": {
            "vertices": data["vertices"],
            "walls": data["walls"],
            "Z": data["Z"],
        }
    }


def getAudienceBounds(audienceArea):
    keys = sorted(
        audienceArea.keys(),
        key=lambda key: int(key[1:])
    )

    points = np.asarray(
        [audienceArea[key] for key in keys],
        dtype=float,
    )

    xMin = float(np.min(points[:, 0]))
    xMax = float(np.max(points[:, 0]))
    yMin = float(np.min(points[:, 1]))
    yMax = float(np.max(points[:, 1]))

    return xMin, xMax, yMin, yMax


def hasMinimumDistance(micPositions, minDistance):
    micPositions = np.asarray(micPositions, dtype=float)

    for i in range(len(micPositions)):
        for j in range(i + 1, len(micPositions)):
            distance = np.linalg.norm(
                micPositions[i, :2] - micPositions[j, :2]
            )

            if distance < minDistance:
                return False

    return True


def computePossibleMicPositions(
    audienceArea,
    nMics=5,
    micHeight=1.2,
    minDistance=0.5,
    nConfigs=200,
    randomSeed=1234,
    margin=0.1,
):
    xMin, xMax, yMin, yMax = getAudienceBounds(audienceArea)

    xMin += margin
    xMax -= margin
    yMin += margin
    yMax -= margin

    if xMin >= xMax or yMin >= yMax:
        raise ValueError("El audience_area queda inválida después de aplicar margin.")

    rng = np.random.default_rng(randomSeed)

    configs = []
    maxTries = nConfigs * 200

    for _ in range(maxTries):
        if len(configs) >= nConfigs:
            break

        xs = rng.uniform(xMin, xMax, size=nMics)
        ys = rng.uniform(yMin, yMax, size=nMics)
        zs = np.full(nMics, micHeight)

        micPositions = np.column_stack([xs, ys, zs])

        if hasMinimumDistance(micPositions, minDistance):
            configs.append([
                tuple(mic)
                for mic in micPositions
            ])

    if len(configs) == 0:
        raise ValueError("No se pudo generar ninguna configuración válida de mics.")

    return configs


def generateCandidateMics(roomParams, roomIndex):
    configs = computePossibleMicPositions(
        audienceArea=roomParams["data"]["audience_area"],
        nMics=1,
        micHeight=MIC_HEIGHT,
        minDistance=0.5,
        nConfigs=N_CANDIDATE_MICS,
        randomSeed=RANDOM_SEED + 1000 * roomIndex,
        margin=AUDIENCE_MARGIN,
    )

    candidateMics = np.asarray(
        [config[0] for config in configs],
        dtype=float,
    )

    return candidateMics


def generateCombos(candidateMics, roomIndex):
    rng = np.random.default_rng(RANDOM_SEED + 1 + 1000 * roomIndex)

    combos = []
    seen = set()
    maxTries = N_COMBOS * 500

    for _ in range(maxTries):
        if len(combos) >= N_COMBOS:
            break

        indices = rng.choice(
            len(candidateMics),
            size=N_MICS_PER_COMBO,
            replace=False,
        )

        indices = tuple(sorted(int(i) for i in indices))

        if indices in seen:
            continue

        micPositions = candidateMics[list(indices)]

        if not hasMinimumDistance(micPositions, MIN_COMBO_DISTANCE):
            continue

        seen.add(indices)
        combos.append(indices)

    if len(combos) < N_COMBOS:
        print(
            f"Warning: solo pude generar {len(combos)} combos "
            f"de {N_COMBOS} pedidos."
        )

    return combos


def ensureReceiversByFreqs(splResponses, freqs):
    splResponses = np.asarray(splResponses, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if splResponses.shape[0] == len(freqs):
        return splResponses.T

    if splResponses.shape[1] == len(freqs):
        return splResponses

    raise ValueError(
        f"No puedo interpretar splResponses shape={splResponses.shape}, "
        f"len(freqs)={len(freqs)}"
    )


def centerCurve(curve):
    curve = np.asarray(curve, dtype=float)
    return curve - np.mean(curve)


def pearsonCorrelation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def rankValues(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)

    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)

    return ranks


def spearmanCorrelation(x, y):
    return pearsonCorrelation(
        rankValues(x),
        rankValues(y),
    )


def impedanceLabel(z):
    z = complex(z)

    if abs(z.imag) < 1e-12:
        label = f"{z.real:g}"
    else:
        label = f"{z.real:g}_{z.imag:g}j"

    return label.replace(".", "p").replace("-", "m")


def saveCsv(rows, path):
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        f.write("sep=;\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def saveJson(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# =========================================================
# Simulación
# =========================================================

def createMesh(roomName, roomParams):
    mesher = Mesher()

    meshPath = mesher.create(
        cleanRoomParams(roomParams),
        room_name=f"{roomName}_{RUN_LABEL}",
        visualize=False,
        source_pos=roomParams["data"]["source_pos"],
    )

    return meshPath


def runDirect(roomName, roomParams, meshPath, candidateMics, order, impedanceValue):
    zLabel = impedanceLabel(impedanceValue)

    caseName = f"{roomName}_order_{order}_z_{zLabel}"
    outputPath = RESPONSES_DIR / roomName / f"{caseName}.npz"
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    if SKIP_COMPLETED and outputPath.exists():
        print(f"Loading existing: {outputPath.name}")
        data = np.load(outputPath)

        return {
            "freqs": np.asarray(data["freqs"], dtype=float),
            "spl": np.asarray(data["spl_responses"], dtype=float),
            "path": outputPath,
        }

    print()
    print("=" * 80)
    print(f"Running room={roomName}, order={order}, z={impedanceValue}")
    print("=" * 80)

    directSimulator = DirectSimulator()

    t0 = time.perf_counter()

    freqsOut, splResponses = directSimulator.simulate(
        mesh_path=meshPath,
        mic_positions=candidateMics,
        order=order,
        room_name=caseName,
        freqs=FREQS,
        use_impedance=True,
        wall_z=impedanceValue,
        floor_z=impedanceValue,
        ceiling_z=impedanceValue,
    )

    runtime = time.perf_counter() - t0

    splResponses = ensureReceiversByFreqs(splResponses, freqsOut)

    np.savez_compressed(
        outputPath,
        freqs=np.asarray(freqsOut),
        spl_responses=np.asarray(splResponses),
        mic_positions=np.asarray(candidateMics),
        source_position=np.asarray(roomParams["data"]["source_pos"]),
        direct_order=np.asarray(order),
        impedance_value=np.asarray(impedanceValue),
        wall_z=np.asarray(impedanceValue),
        floor_z=np.asarray(impedanceValue),
        ceiling_z=np.asarray(impedanceValue),
        simulation_time_s=np.asarray(runtime),
    )

    print(f"Saved: {outputPath}")
    print(f"Runtime: {runtime:.2f} s")

    return {
        "freqs": np.asarray(freqsOut, dtype=float),
        "spl": np.asarray(splResponses, dtype=float),
        "path": outputPath,
    }


# =========================================================
# Evaluación
# =========================================================

def meanMicCorrelation(order1Spl, order2Spl):
    nMics = min(order1Spl.shape[0], order2Spl.shape[0])

    corrs = []

    for micIndex in range(nMics):
        c1 = centerCurve(order1Spl[micIndex])
        c2 = centerCurve(order2Spl[micIndex])

        corrs.append(pearsonCorrelation(c1, c2))

    return float(np.nanmean(corrs)), corrs


def evaluateCombos(order1Spl, order2Spl, combos):
    evaluator = Evaluator()

    comboRows = []

    for comboIndex, combo in enumerate(combos):
        combo = list(combo)

        response1 = order1Spl[combo, :]
        response2 = order2Spl[combo, :]

        msfd1 = evaluator.evaluate_msfd(
            response=response1,
            input_is_db=True,
            weight_magnitude=0.5,
            weight_spatial=0.5,
        )["MSFD"]

        msfd2 = evaluator.evaluate_msfd(
            response=response2,
            input_is_db=True,
            weight_magnitude=0.5,
            weight_spatial=0.5,
        )["MSFD"]

        comboRows.append({
            "combo_index": comboIndex,
            "combo_indices": ",".join(str(i) for i in combo),
            "MSFD_order_1": float(msfd1),
            "MSFD_order_2": float(msfd2),
            "MSFD_abs_error": float(abs(msfd1 - msfd2)),
            "MSFD_signed_error": float(msfd2 - msfd1),
        })

    return comboRows


def topKOverlap(comboRows, k):
    order1Top = sorted(comboRows, key=lambda row: row["MSFD_order_1"])[:k]
    order2Top = sorted(comboRows, key=lambda row: row["MSFD_order_2"])[:k]

    top1 = {row["combo_index"] for row in order1Top}
    top2 = {row["combo_index"] for row in order2Top}

    return len(top1.intersection(top2))


def summarizeRoomImpedance(roomName, impedanceValue, order1Result, order2Result, combos):
    if not np.allclose(order1Result["freqs"], order2Result["freqs"]):
        raise ValueError("Las frecuencias de orden 1 y orden 2 no coinciden.")

    meanCorr, micCorrs = meanMicCorrelation(
        order1Spl=order1Result["spl"],
        order2Spl=order2Result["spl"],
    )

    comboRows = evaluateCombos(
        order1Spl=order1Result["spl"],
        order2Spl=order2Result["spl"],
        combos=combos,
    )

    msfd1 = [row["MSFD_order_1"] for row in comboRows]
    msfd2 = [row["MSFD_order_2"] for row in comboRows]

    pearson = pearsonCorrelation(msfd1, msfd2)
    spearman = spearmanCorrelation(msfd1, msfd2)

    summary = {
        "room_name": roomName,
        "impedance": str(complex(impedanceValue)),
        "n_candidate_mics": int(N_CANDIDATE_MICS),
        "n_combos": int(len(combos)),
        "n_mics_per_combo": int(N_MICS_PER_COMBO),
        "mean_mic_curve_corr": float(meanCorr),
        "min_mic_curve_corr": float(np.nanmin(micCorrs)),
        "max_mic_curve_corr": float(np.nanmax(micCorrs)),
        "msfd_pearson": float(pearson),
        "msfd_spearman": float(spearman),
        "msfd_mean_abs_error": float(np.mean([row["MSFD_abs_error"] for row in comboRows])),
        "top5_overlap": int(topKOverlap(comboRows, 5)),
        "top10_overlap": int(topKOverlap(comboRows, 10)),
        "order1_file": str(order1Result["path"]),
        "order2_file": str(order2Result["path"]),
    }

    return summary, comboRows


def summarizeGlobal(summaryRows):
    byImpedance = {}

    for row in summaryRows:
        z = row["impedance"]

        if z not in byImpedance:
            byImpedance[z] = []

        byImpedance[z].append(row)

    globalRows = []

    for z, rows in byImpedance.items():
        globalRows.append({
            "room_name": "GLOBAL_MEAN",
            "impedance": z,
            "n_rooms": len(rows),
            "mean_mic_curve_corr": float(np.mean([r["mean_mic_curve_corr"] for r in rows])),
            "msfd_pearson": float(np.mean([r["msfd_pearson"] for r in rows])),
            "msfd_spearman": float(np.mean([r["msfd_spearman"] for r in rows])),
            "top5_overlap": float(np.mean([r["top5_overlap"] for r in rows])),
            "top10_overlap": float(np.mean([r["top10_overlap"] for r in rows])),
            "msfd_mean_abs_error": float(np.mean([r["msfd_mean_abs_error"] for r in rows])),
        })

    return globalRows


# =========================================================
# Main
# =========================================================

def main():
    print()
    print("=" * 80)
    print("Direct order 1 vs order 2 - many mic candidates - multiroom")
    print("=" * 80)

    print(f"Rooms: {len(ROOMS)}")
    print(f"Candidate mics per room: {N_CANDIDATE_MICS}")
    print(f"Combos per room: {N_COMBOS}")
    print(f"Freqs: {FREQS[0]} Hz to {FREQS[-1]} Hz | n={len(FREQS)}")
    print(f"Impedances: {IMPEDANCE_VALUES}")

    summaryRows = []
    allComboRows = []

    globalStart = time.perf_counter()

    for roomIndex, (roomName, roomParams) in enumerate(ROOMS.items(), start=1):
        print()
        print("#" * 80)
        print(f"Room [{roomIndex}/{len(ROOMS)}]: {roomName}")
        print("#" * 80)

        roomStart = time.perf_counter()

        candidateMics = generateCandidateMics(roomParams, roomIndex)
        combos = generateCombos(candidateMics, roomIndex)

        roomMetaPath = OUT_DIR / f"{roomName}_mic_candidates_and_combos.npz"

        np.savez_compressed(
            roomMetaPath,
            mic_positions=np.asarray(candidateMics),
            combos=np.asarray(combos, dtype=int),
            source_position=np.asarray(roomParams["data"]["source_pos"]),
        )

        meshPath = createMesh(roomName, roomParams)

        for impedanceValue in IMPEDANCE_VALUES:
            orderResults = {}

            for order in DIRECT_ORDERS:
                orderResults[order] = runDirect(
                    roomName=roomName,
                    roomParams=roomParams,
                    meshPath=meshPath,
                    candidateMics=candidateMics,
                    order=order,
                    impedanceValue=impedanceValue,
                )

            summary, comboRows = summarizeRoomImpedance(
                roomName=roomName,
                impedanceValue=impedanceValue,
                order1Result=orderResults[1],
                order2Result=orderResults[2],
                combos=combos,
            )

            summaryRows.append(summary)

            for row in comboRows:
                row["room_name"] = roomName
                row["impedance"] = str(complex(impedanceValue))
                allComboRows.append(row)

            print()
            print("-" * 80)
            print(f"room={roomName}, z={impedanceValue}")
            print(f"  mean mic corr: {summary['mean_mic_curve_corr']:.3f}")
            print(f"  MSFD Pearson: {summary['msfd_pearson']:.3f}")
            print(f"  MSFD Spearman: {summary['msfd_spearman']:.3f}")
            print(f"  Top10 overlap: {summary['top10_overlap']} / 10")

        roomTime = time.perf_counter() - roomStart
        print(f"Room runtime: {roomTime:.2f} s")

    globalRows = summarizeGlobal(summaryRows)

    saveCsv(summaryRows + globalRows, SUMMARY_CSV)
    saveCsv(allComboRows, COMBOS_CSV)

    totalRuntime = time.perf_counter() - globalStart

    saveJson(
        {
            "run_label": RUN_LABEL,
            "freqs": {
                "min": float(FREQS[0]),
                "max": float(FREQS[-1]),
                "n": int(len(FREQS)),
            },
            "n_rooms": int(len(ROOMS)),
            "n_candidate_mics": int(N_CANDIDATE_MICS),
            "n_combos": int(N_COMBOS),
            "n_mics_per_combo": int(N_MICS_PER_COMBO),
            "impedance_values": [str(complex(z)) for z in IMPEDANCE_VALUES],
            "summary": summaryRows,
            "global_summary": globalRows,
            "total_runtime_s": float(totalRuntime),
        },
        SUMMARY_JSON,
    )

    print()
    print("=" * 80)
    print("Finished.")
    print(f"Output dir: {OUT_DIR}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Combos CSV: {COMBOS_CSV}")
    print(f"Total runtime: {totalRuntime:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()