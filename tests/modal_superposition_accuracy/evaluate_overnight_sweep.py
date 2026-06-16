import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from room_modal_optimizer.evaluation.evaluator import ModalEvaluator


# =========================================================
# Paths
# =========================================================

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

SWEEP_DIR = RESULTS_DIR / "overnight_sweep"

DIRECT_DIR = SWEEP_DIR / "direct"
MODAL_DIR = SWEEP_DIR / "modal"

OUTPUT_DIR = SWEEP_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_SUMMARY_CSV = OUTPUT_DIR / "pair_summary.csv"
PAIR_CASES_CSV = OUTPUT_DIR / "pair_cases.csv"
PAIR_SUMMARY_JSON = OUTPUT_DIR / "pair_summary.json"


# =========================================================
# Helpers generales
# =========================================================

def formatValueForExcel(value):
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.10f}".replace(".", ",")

    if isinstance(value, (int, np.integer)):
        return int(value)

    return value


def saveCsv(rows, outputPath):
    if len(rows) == 0:
        print(f"No hay filas para guardar en {outputPath}")
        return

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    formattedRows = [
        {
            key: formatValueForExcel(value)
            for key, value in row.items()
        }
        for row in rows
    ]

    with open(outputPath, "w", newline="", encoding="utf-8-sig") as f:
        f.write("sep=;\n")

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(formattedRows)

    print(f"CSV guardado: {outputPath}")


def saveJson(data, outputPath):
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"JSON guardado: {outputPath}")


# =========================================================
# Parseo de nombres
# =========================================================

def parseDirectPath(npzPath):
    """
    Ejemplo:
        single_C001_direct_impedance_step_0.5.npz
        single_C001_direct_rigid_step_1.0.npz
    """
    pattern = r"^(?P<base>.*?_C\d+)_direct_(?P<direct_model>rigid|impedance)_step_(?P<step>[0-9.]+)$"

    match = re.match(pattern, npzPath.stem)

    if match is None:
        raise ValueError(f"No pude parsear directo: {npzPath.name}")

    return {
        "base_case_name": match.group("base"),
        "direct_model": match.group("direct_model"),
        "freq_step": float(match.group("step")),
        "path": npzPath,
    }


def parseModalPath(npzPath):
    """
    Ejemplo:
        single_C001_modal_zeta_0.005_step_1.0.npz
    """
    pattern = r"^(?P<base>.*?_C\d+)_modal_zeta_(?P<zeta>[0-9.]+)_step_(?P<step>[0-9.]+)$"

    match = re.match(pattern, npzPath.stem)

    if match is None:
        raise ValueError(f"No pude parsear modal: {npzPath.name}")

    return {
        "base_case_name": match.group("base"),
        "zeta": float(match.group("zeta")),
        "freq_step": float(match.group("step")),
        "path": npzPath,
    }


# =========================================================
# Señales / métricas
# =========================================================

def ensureReceiversByFreqs(splResponses, freqs):
    """
    Devuelve siempre:
        [n_mics, n_freqs]
    """
    splResponses = np.asarray(splResponses, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if splResponses.ndim != 2:
        raise ValueError(f"spl_responses debe ser 2D. Shape recibido: {splResponses.shape}")

    nFreqs = len(freqs)

    if splResponses.shape[0] == nFreqs:
        return splResponses.T

    if splResponses.shape[1] == nFreqs:
        return splResponses

    raise ValueError(
        f"No puedo inferir orientación. "
        f"spl_responses.shape={splResponses.shape}, n_freqs={nFreqs}"
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
    """
    Menor valor = mejor ranking.
    """
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


def evaluateResponse(freqs, splResponses):
    responseDb = ensureReceiversByFreqs(
        splResponses=splResponses,
        freqs=freqs,
    )

    result = ModalEvaluator.evaluate_msfd(
        response=responseDb,
        input_is_db=True,
        weight_magnitude=0.5,
        weight_spatial=0.5,
    )

    return responseDb, {
        "MSFD": float(result["MSFD"]),
        "MD": float(result["MD"]),
        "SD": float(result["SD"]),
    }


def loadEvaluatedNpz(npzPath):
    data = np.load(npzPath)

    freqs = np.asarray(data["freqs"], dtype=float)
    splResponses = np.asarray(data["spl_responses"], dtype=float)

    responseDb, metrics = evaluateResponse(
        freqs=freqs,
        splResponses=splResponses,
    )

    return {
        "path": npzPath,
        "freqs": freqs,
        "response_db": responseDb,
        "MSFD": metrics["MSFD"],
        "MD": metrics["MD"],
        "SD": metrics["SD"],
    }


def caseCurveMetrics(directResponse, modalResponse):
    """
    Compara una config Cxxx.

    Usa la curva promedio entre mics, centrada, como en test_new_modal.
    Además calcula correlación promedio por mic.
    """
    directResponse = np.asarray(directResponse, dtype=float)
    modalResponse = np.asarray(modalResponse, dtype=float)

    directMean = np.mean(directResponse, axis=0)
    modalMean = np.mean(modalResponse, axis=0)

    directMeanCentered = centerCurve(directMean)
    modalMeanCentered = centerCurve(modalMean)

    meanCorr = pearsonCorrelation(
        directMeanCentered,
        modalMeanCentered,
    )

    meanMae = float(np.mean(np.abs(directMeanCentered - modalMeanCentered)))

    nMics = min(directResponse.shape[0], modalResponse.shape[0])

    micCorrs = []
    micMaes = []

    for micIndex in range(nMics):
        directMicCentered = centerCurve(directResponse[micIndex])
        modalMicCentered = centerCurve(modalResponse[micIndex])

        micCorrs.append(
            pearsonCorrelation(
                directMicCentered,
                modalMicCentered,
            )
        )

        micMaes.append(
            float(np.mean(np.abs(directMicCentered - modalMicCentered)))
        )

    return {
        "mean_curve_corr": float(meanCorr),
        "mean_curve_centered_mae": float(meanMae),
        "mic_curve_corr_mean": float(np.nanmean(micCorrs)),
        "mic_curve_centered_mae_mean": float(np.nanmean(micMaes)),
    }


# =========================================================
# Comparación ranking
# =========================================================

def addRanks(rows, metric):
    directValues = [row[f"{metric}_direct"] for row in rows]
    modalValues = [row[f"{metric}_modal"] for row in rows]

    directRanks = rankValues(directValues)
    modalRanks = rankValues(modalValues)

    for row, directRank, modalRank in zip(rows, directRanks, modalRanks):
        row[f"rank_{metric}_direct"] = int(directRank)
        row[f"rank_{metric}_modal"] = int(modalRank)
        row[f"rank_{metric}_abs_error"] = int(abs(directRank - modalRank))
        row[f"rank_{metric}_signed_error"] = int(modalRank - directRank)


def topKOverlap(rows, metric, k):
    directSorted = sorted(rows, key=lambda row: row[f"{metric}_direct"])
    modalSorted = sorted(rows, key=lambda row: row[f"{metric}_modal"])

    directTop = {row["base_case_name"] for row in directSorted[:k]}
    modalTop = {row["base_case_name"] for row in modalSorted[:k]}

    overlap = directTop.intersection(modalTop)

    return {
        "count": int(len(overlap)),
        "ratio": float(len(overlap) / k),
        "cases": sorted(overlap),
    }


def summarizePair(caseRows):
    for metric in ["MSFD", "MD", "SD"]:
        addRanks(caseRows, metric)

    first = caseRows[0]

    summary = {
        "freq_step": first["freq_step"],
        "direct_model": first["direct_model"],
        "use_impedance": first["direct_model"] == "impedance",
        "zeta": first["zeta"],
        "n_common_cases": int(len(caseRows)),

        "mean_curve_corr_avg": float(np.nanmean([row["mean_curve_corr"] for row in caseRows])),
        "mean_curve_corr_median": float(np.nanmedian([row["mean_curve_corr"] for row in caseRows])),
        "mean_curve_centered_mae_avg": float(np.nanmean([row["mean_curve_centered_mae"] for row in caseRows])),

        "mic_curve_corr_avg": float(np.nanmean([row["mic_curve_corr_mean"] for row in caseRows])),
        "mic_curve_centered_mae_avg": float(np.nanmean([row["mic_curve_centered_mae_mean"] for row in caseRows])),
    }

    for metric in ["MSFD", "MD", "SD"]:
        directValues = [row[f"{metric}_direct"] for row in caseRows]
        modalValues = [row[f"{metric}_modal"] for row in caseRows]

        absErrors = [row[f"{metric}_abs_error"] for row in caseRows]
        rankAbsErrors = [row[f"rank_{metric}_abs_error"] for row in caseRows]

        top5 = topKOverlap(caseRows, metric, 5)
        top10 = topKOverlap(caseRows, metric, 10)
        top20 = topKOverlap(caseRows, metric, 20)

        metricLower = metric.lower()

        summary[f"{metricLower}_pearson"] = pearsonCorrelation(directValues, modalValues)
        summary[f"{metricLower}_spearman"] = spearmanCorrelation(directValues, modalValues)
        summary[f"{metricLower}_mean_abs_error"] = float(np.mean(absErrors))
        summary[f"{metricLower}_median_abs_error"] = float(np.median(absErrors))
        summary[f"{metricLower}_mean_rank_abs_error"] = float(np.mean(rankAbsErrors))
        summary[f"{metricLower}_median_rank_abs_error"] = float(np.median(rankAbsErrors))

        summary[f"{metricLower}_top5_overlap"] = top5["count"]
        summary[f"{metricLower}_top5_ratio"] = top5["ratio"]
        summary[f"{metricLower}_top10_overlap"] = top10["count"]
        summary[f"{metricLower}_top10_ratio"] = top10["ratio"]
        summary[f"{metricLower}_top20_overlap"] = top20["count"]
        summary[f"{metricLower}_top20_ratio"] = top20["ratio"]

    return summary


# =========================================================
# Indexado de archivos
# =========================================================

def buildDirectIndex():
    directIndex = {}

    for npzPath in sorted(DIRECT_DIR.glob("*.npz")):
        parsed = parseDirectPath(npzPath)

        key = (
            parsed["freq_step"],
            parsed["direct_model"],
            parsed["base_case_name"],
        )

        directIndex[key] = parsed["path"]

    return directIndex


def buildModalIndex():
    modalIndex = {}

    for npzPath in sorted(MODAL_DIR.glob("*.npz")):
        parsed = parseModalPath(npzPath)

        key = (
            parsed["freq_step"],
            parsed["zeta"],
            parsed["base_case_name"],
        )

        modalIndex[key] = parsed["path"]

    return modalIndex


def getAvailableDirectPairs(directIndex):
    return sorted({
        (freqStep, directModel)
        for freqStep, directModel, baseCaseName in directIndex.keys()
    })


def getAvailableModalPairs(modalIndex):
    return sorted({
        (freqStep, zeta)
        for freqStep, zeta, baseCaseName in modalIndex.keys()
    })


# =========================================================
# Main
# =========================================================

def main():
    directIndex = buildDirectIndex()
    modalIndex = buildModalIndex()

    directPairs = getAvailableDirectPairs(directIndex)
    modalPairs = getAvailableModalPairs(modalIndex)

    print()
    print("=" * 80)
    print("Evaluando overnight sweep")
    print("=" * 80)
    print(f"Direct files: {len(directIndex)}")
    print(f"Modal files: {len(modalIndex)}")
    print(f"Direct variants: {directPairs}")
    print(f"Modal variants: {modalPairs}")

    allCaseRows = []
    summaryRows = []
    summaryJson = []

    for freqStep, directModel in directPairs:
        modalPairsSameStep = [
            (modalFreqStep, zeta)
            for modalFreqStep, zeta in modalPairs
            if modalFreqStep == freqStep
        ]

        directCases = {
            baseCaseName
            for directFreqStep, directModelName, baseCaseName in directIndex.keys()
            if directFreqStep == freqStep and directModelName == directModel
        }

        for _, zeta in modalPairsSameStep:
            modalCases = {
                baseCaseName
                for modalFreqStep, modalZeta, baseCaseName in modalIndex.keys()
                if modalFreqStep == freqStep and modalZeta == zeta
            }

            commonCases = sorted(directCases.intersection(modalCases))

            if len(commonCases) == 0:
                continue

            print()
            print("-" * 80)
            print(
                f"Comparando step={freqStep}, "
                f"direct={directModel}, "
                f"modal zeta={zeta}, "
                f"cases={len(commonCases)}"
            )

            pairCaseRows = []

            for baseCaseName in commonCases:
                directPath = directIndex[(freqStep, directModel, baseCaseName)]
                modalPath = modalIndex[(freqStep, zeta, baseCaseName)]

                directData = loadEvaluatedNpz(directPath)
                modalData = loadEvaluatedNpz(modalPath)

                if not np.allclose(directData["freqs"], modalData["freqs"]):
                    raise ValueError(
                        f"Frecuencias distintas para {baseCaseName}: "
                        f"{directPath.name} vs {modalPath.name}"
                    )

                curveMetrics = caseCurveMetrics(
                    directResponse=directData["response_db"],
                    modalResponse=modalData["response_db"],
                )

                row = {
                    "base_case_name": baseCaseName,
                    "freq_step": freqStep,
                    "direct_model": directModel,
                    "zeta": zeta,
                    "direct_file": directPath.name,
                    "modal_file": modalPath.name,

                    **curveMetrics,

                    "MSFD_direct": directData["MSFD"],
                    "MSFD_modal": modalData["MSFD"],
                    "MSFD_abs_error": abs(directData["MSFD"] - modalData["MSFD"]),
                    "MSFD_signed_error": modalData["MSFD"] - directData["MSFD"],

                    "MD_direct": directData["MD"],
                    "MD_modal": modalData["MD"],
                    "MD_abs_error": abs(directData["MD"] - modalData["MD"]),
                    "MD_signed_error": modalData["MD"] - directData["MD"],

                    "SD_direct": directData["SD"],
                    "SD_modal": modalData["SD"],
                    "SD_abs_error": abs(directData["SD"] - modalData["SD"]),
                    "SD_signed_error": modalData["SD"] - directData["SD"],
                }

                pairCaseRows.append(row)

            pairSummary = summarizePair(pairCaseRows)

            allCaseRows.extend(pairCaseRows)
            summaryRows.append(pairSummary)
            summaryJson.append(pairSummary)

            print(
                f"  MSFD Spearman={pairSummary['msfd_spearman']:.3f}, "
                f"Top10={pairSummary['msfd_top10_overlap']}/10, "
                f"Mean curve corr={pairSummary['mean_curve_corr_avg']:.3f}"
            )

    summaryRows = sorted(
        summaryRows,
        key=lambda row: (
            -row["msfd_spearman"],
            -row["msfd_top10_overlap"],
            -row["mean_curve_corr_avg"],
        ),
    )

    saveCsv(summaryRows, PAIR_SUMMARY_CSV)
    saveCsv(allCaseRows, PAIR_CASES_CSV)
    saveJson(summaryJson, PAIR_SUMMARY_JSON)

    print()
    print("=" * 80)
    print("Top configuraciones por MSFD Spearman")
    print("=" * 80)

    for row in summaryRows[:10]:
        print(
            f"step={row['freq_step']}, "
            f"direct={row['direct_model']}, "
            f"zeta={row['zeta']} | "
            f"MSFD Spearman={row['msfd_spearman']:.3f}, "
            f"Top10={row['msfd_top10_overlap']}/10, "
            f"Mean curve corr={row['mean_curve_corr_avg']:.3f}"
        )


if __name__ == "__main__":
    main()