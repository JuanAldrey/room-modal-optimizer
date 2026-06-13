import csv
import json
import math
from pathlib import Path

import numpy as np

from room_modal_optimizer.evaluation.modal_evaluator import ModalEvaluator


# =========================================================
# Paths
# =========================================================

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

DIRECT_RUN_LABEL = "single_2_direct_order_1_2_no_Z"
MODAL_RUN_LABEL = "single_2_modal_order_2_ZETA_0.0001_2"

DIRECT_RESULTS_DIR = RESULTS_DIR / DIRECT_RUN_LABEL
MODAL_RESULTS_DIR = RESULTS_DIR / MODAL_RUN_LABEL

OUTPUT_DIR = RESULTS_DIR / "msfd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIRECT_MSFD_CSV = OUTPUT_DIR / f"{DIRECT_RUN_LABEL}_msfd.csv"
MODAL_MSFD_CSV = OUTPUT_DIR / f"{MODAL_RUN_LABEL}_msfd.csv"
COMPARISON_CSV = OUTPUT_DIR / "msfd_direct_vs_modal.csv"
COMPARISON_JSON = OUTPUT_DIR / "msfd_comparison_summary.json"


# =========================================================
# Helpers
# =========================================================

def getBaseCaseName(npzPath):
    """
    Convierte, por ejemplo:

    room_04w_01_C1_direct_order_2.npz
    room_04w_01_C1_modal_order_2.npz

    en:

    room_04w_01_C1
    """
    parts = npzPath.stem.split("_")

    # room_04w_01_C1_...
    return "_".join(parts[:4])


def ensureReceiversByFreqs(splResponses, freqs):
    """
    Devuelve siempre shape:

    [n_receivers, n_freqs]
    """
    splResponses = np.asarray(splResponses, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if splResponses.ndim != 2:
        raise ValueError(f"spl_responses debe ser 2D. Shape recibido: {splResponses.shape}")

    nFreqs = len(freqs)

    if splResponses.shape[0] == nFreqs:
        # Caso directo típico: [n_freqs, n_mics]
        return splResponses.T

    if splResponses.shape[1] == nFreqs:
        # Caso modal típico: [n_mics, n_freqs]
        return splResponses

    raise ValueError(
        f"No puedo inferir orientación. spl_responses.shape={splResponses.shape}, n_freqs={nFreqs}"
    )


def evaluateNpzFile(npzPath):
    data = np.load(npzPath)

    freqs = data["freqs"]
    splResponses = data["spl_responses"]

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

    baseCaseName = getBaseCaseName(npzPath)

    return {
        "base_case_name": baseCaseName,
        "file_name": npzPath.name,
        "file_path": str(npzPath),
        "MSFD": result["MSFD"],
        "MD": result["MD"],
        "SD": result["SD"],
        "n_freqs": int(len(freqs)),
        "n_mics": int(responseDb.shape[0]),
        "freq_min": float(np.min(freqs)),
        "freq_max": float(np.max(freqs)),
    }


def evaluateResultsDirectory(resultsDir, outputCsv):
    rows = []

    npzFiles = sorted(resultsDir.glob("*.npz"))

    for npzPath in npzFiles:
        # Ignora archivos auxiliares tipo room_modal.npz
        if "room_modal" in npzPath.stem:
            continue

        try:
            row = evaluateNpzFile(npzPath)
            rows.append(row)
            print(f"OK: {npzPath.name} -> MSFD={row['MSFD']:.3f}")

        except Exception as e:
            print(f"ERROR: {npzPath.name}: {e}")

    rows = sorted(rows, key=lambda row: row["base_case_name"])

    saveCsv(rows, outputCsv)

    return rows


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

    formattedRows = []

    for row in rows:
        formattedRow = {
            key: formatValueForExcel(value)
            for key, value in row.items()
        }

        formattedRows.append(formattedRow)

    with open(outputPath, "w", newline="", encoding="utf-8-sig") as f:
        f.write("sep=;\n")

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=";"
        )

        writer.writeheader()
        writer.writerows(formattedRows)

    print(f"CSV guardado: {outputPath}")


def rankValues(values):
    """
    Ranking simple sin scipy.
    Menor MSFD = mejor ranking.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)

    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)

    return ranks


def pearsonCorrelation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def spearmanCorrelation(x, y):
    rankX = rankValues(x)
    rankY = rankValues(y)

    return pearsonCorrelation(rankX, rankY)


def topKOverlapByMetric(directRows, modalRows, metric="MSFD", k=5):
    """
    Menor métrica = mejor ranking.
    metric puede ser: MSFD, MD o SD.
    """
    directSorted = sorted(directRows, key=lambda row: row[metric])
    modalSorted = sorted(modalRows, key=lambda row: row[metric])

    directTop = {row["base_case_name"] for row in directSorted[:k]}
    modalTop = {row["base_case_name"] for row in modalSorted[:k]}

    overlap = directTop.intersection(modalTop)

    return {
        "metric": metric,
        "k": int(k),
        "direct_top": sorted(directTop),
        "modal_top": sorted(modalTop),
        "overlap": sorted(overlap),
        "overlap_count": int(len(overlap)),
        "overlap_ratio": float(len(overlap) / k),
    }


def addRanksToComparisonRows(comparisonRows, metric):
    """
    Agrega rankings direct/modal para una métrica.
    Menor valor = mejor ranking.
    """
    directValues = [row[f"{metric}_direct"] for row in comparisonRows]
    modalValues = [row[f"{metric}_modal"] for row in comparisonRows]

    directRanks = rankValues(directValues)
    modalRanks = rankValues(modalValues)

    for row, directRank, modalRank in zip(comparisonRows, directRanks, modalRanks):
        row[f"rank_{metric}_direct"] = int(directRank)
        row[f"rank_{metric}_modal"] = int(modalRank)
        row[f"rank_{metric}_abs_error"] = int(abs(directRank - modalRank))
        row[f"rank_{metric}_signed_error"] = int(modalRank - directRank)


def compareDirectAndModal(directRows, modalRows):
    directByCase = {
        row["base_case_name"]: row
        for row in directRows
    }

    modalByCase = {
        row["base_case_name"]: row
        for row in modalRows
    }

    commonCases = sorted(
        set(directByCase.keys()).intersection(set(modalByCase.keys()))
    )

    comparisonRows = []

    for caseName in commonCases:
        direct = directByCase[caseName]
        modal = modalByCase[caseName]

        comparisonRows.append({
            "base_case_name": caseName,

            "MSFD_direct": direct["MSFD"],
            "MSFD_modal": modal["MSFD"],
            "MSFD_abs_error": abs(direct["MSFD"] - modal["MSFD"]),
            "MSFD_signed_error": modal["MSFD"] - direct["MSFD"],

            "MD_direct": direct["MD"],
            "MD_modal": modal["MD"],
            "MD_abs_error": abs(direct["MD"] - modal["MD"]),
            "MD_signed_error": modal["MD"] - direct["MD"],

            "SD_direct": direct["SD"],
            "SD_modal": modal["SD"],
            "SD_abs_error": abs(direct["SD"] - modal["SD"]),
            "SD_signed_error": modal["SD"] - direct["SD"],
        })

    # Agrega rankings por métrica
    addRanksToComparisonRows(comparisonRows, "MSFD")
    addRanksToComparisonRows(comparisonRows, "MD")
    addRanksToComparisonRows(comparisonRows, "SD")

    # Guardar CSV comparativo completo
    saveCsv(comparisonRows, COMPARISON_CSV)

    directMsfd = [row["MSFD_direct"] for row in comparisonRows]
    modalMsfd = [row["MSFD_modal"] for row in comparisonRows]

    directMd = [row["MD_direct"] for row in comparisonRows]
    modalMd = [row["MD_modal"] for row in comparisonRows]

    directSd = [row["SD_direct"] for row in comparisonRows]
    modalSd = [row["SD_modal"] for row in comparisonRows]

    summary = {
        "n_common_cases": int(len(commonCases)),

        "pearson_msfd": pearsonCorrelation(directMsfd, modalMsfd),
        "spearman_msfd": spearmanCorrelation(directMsfd, modalMsfd),
        "mean_abs_error_msfd": float(np.mean([row["MSFD_abs_error"] for row in comparisonRows])),
        "median_abs_error_msfd": float(np.median([row["MSFD_abs_error"] for row in comparisonRows])),
        "mean_rank_abs_error_msfd": float(np.mean([row["rank_MSFD_abs_error"] for row in comparisonRows])),
        "median_rank_abs_error_msfd": float(np.median([row["rank_MSFD_abs_error"] for row in comparisonRows])),

        "pearson_md": pearsonCorrelation(directMd, modalMd),
        "spearman_md": spearmanCorrelation(directMd, modalMd),
        "mean_abs_error_md": float(np.mean([row["MD_abs_error"] for row in comparisonRows])),
        "median_abs_error_md": float(np.median([row["MD_abs_error"] for row in comparisonRows])),
        "mean_rank_abs_error_md": float(np.mean([row["rank_MD_abs_error"] for row in comparisonRows])),
        "median_rank_abs_error_md": float(np.median([row["rank_MD_abs_error"] for row in comparisonRows])),

        "pearson_sd": pearsonCorrelation(directSd, modalSd),
        "spearman_sd": spearmanCorrelation(directSd, modalSd),
        "mean_abs_error_sd": float(np.mean([row["SD_abs_error"] for row in comparisonRows])),
        "median_abs_error_sd": float(np.median([row["SD_abs_error"] for row in comparisonRows])),
        "mean_rank_abs_error_sd": float(np.mean([row["rank_SD_abs_error"] for row in comparisonRows])),
        "median_rank_abs_error_sd": float(np.median([row["rank_SD_abs_error"] for row in comparisonRows])),

        "top_5_overlap_msfd": topKOverlapByMetric(directRows, modalRows, metric="MSFD", k=5),
        "top_5_overlap_md": topKOverlapByMetric(directRows, modalRows, metric="MD", k=5),
        "top_5_overlap_sd": topKOverlapByMetric(directRows, modalRows, metric="SD", k=5),

        "top_10_overlap_msfd": topKOverlapByMetric(directRows, modalRows, metric="MSFD", k=10),
        "top_10_overlap_md": topKOverlapByMetric(directRows, modalRows, metric="MD", k=10),
        "top_10_overlap_sd": topKOverlapByMetric(directRows, modalRows, metric="SD", k=10),
    }

    with open(COMPARISON_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Comparison CSV guardado: {COMPARISON_CSV}")
    print(f"Comparison summary guardado: {COMPARISON_JSON}")

    print()
    print("Resumen comparación:")
    print(f"Casos comunes: {summary['n_common_cases']}")

    print()
    print("MSFD:")
    print(f"  Pearson: {summary['pearson_msfd']:.3f}")
    print(f"  Spearman: {summary['spearman_msfd']:.3f}")
    print(f"  Error medio absoluto: {summary['mean_abs_error_msfd']:.3f}")
    print(f"  Error medio ranking: {summary['mean_rank_abs_error_msfd']:.3f}")
    print(f"  Top 5 overlap: {summary['top_5_overlap_msfd']['overlap_count']} / 5")
    print(f"  Top 10 overlap: {summary['top_10_overlap_msfd']['overlap_count']} / 10")

    print()
    print("MD:")
    print(f"  Pearson: {summary['pearson_md']:.3f}")
    print(f"  Spearman: {summary['spearman_md']:.3f}")
    print(f"  Error medio absoluto: {summary['mean_abs_error_md']:.3f}")
    print(f"  Error medio ranking: {summary['mean_rank_abs_error_md']:.3f}")
    print(f"  Top 5 overlap: {summary['top_5_overlap_md']['overlap_count']} / 5")
    print(f"  Top 10 overlap: {summary['top_10_overlap_md']['overlap_count']} / 10")

    print()
    print("SD:")
    print(f"  Pearson: {summary['pearson_sd']:.3f}")
    print(f"  Spearman: {summary['spearman_sd']:.3f}")
    print(f"  Error medio absoluto: {summary['mean_abs_error_sd']:.3f}")
    print(f"  Error medio ranking: {summary['mean_rank_abs_error_sd']:.3f}")
    print(f"  Top 5 overlap: {summary['top_5_overlap_sd']['overlap_count']} / 5")
    print(f"  Top 10 overlap: {summary['top_10_overlap_sd']['overlap_count']} / 10")

    return comparisonRows, summary


# =========================================================
# Main
# =========================================================

def main():
    directRows = []

    if DIRECT_RESULTS_DIR.exists():
        print()
        print("=" * 80)
        print("Evaluando MSFD directo")
        print("=" * 80)

        directRows = evaluateResultsDirectory(
            resultsDir=DIRECT_RESULTS_DIR,
            outputCsv=DIRECT_MSFD_CSV,
        )
    else:
        print(f"No existe carpeta directa: {DIRECT_RESULTS_DIR}")

    modalRows = []

    if MODAL_RESULTS_DIR.exists():
        print()
        print("=" * 80)
        print("Evaluando MSFD modal")
        print("=" * 80)

        modalRows = evaluateResultsDirectory(
            resultsDir=MODAL_RESULTS_DIR,
            outputCsv=MODAL_MSFD_CSV,
        )
    else:
        print(f"No existe carpeta modal: {MODAL_RESULTS_DIR}")

    if len(directRows) > 0 and len(modalRows) > 0:
        print()
        print("=" * 80)
        print("Comparando directo vs modal")
        print("=" * 80)

        compareDirectAndModal(
            directRows=directRows,
            modalRows=modalRows,
        )


if __name__ == "__main__":
    main()