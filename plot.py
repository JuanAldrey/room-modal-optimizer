from room_modal_optimizer.pipeline.pipeline import Pipeline
from room_modal_optimizer.meshing.mesher import Mesher
from room_modal_optimizer.simulation.direct_simulator import DirectSimulator
from room_modal_optimizer.evaluation.evaluator import Evaluator

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# Room params
# =========================================================

baseParams = {
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

bruteforceParams = {
    "data": {
        "vertices": {
            "V1": [-2.75, 0.25],
            "V2": [ 2.75, 0.25],
            "V3": [ 2.25, 3.75],
            "V4": [-2.25, 3.75],
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
        "Z": 3.6,
        "source_pos": [[0.0, 3.0, 1.5]],
    }
}

gaParams = {
    "data": {
        "vertices": {
            "V1": [-2.9086627578310877, 0.47831999461950336],
            "V2": [ 2.9086627578310877, 0.47831999461950336],
            "V3": [ 2.0944326910433904, 4.2909794684711695],
            "V4": [-2.0944326910433904, 4.2909794684711695],
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
        "Z": 3.534910033007663,
        "source_pos": [[0.0, 3.0, 1.5]],
    }
}


# =========================================================
# Results
# =========================================================

baseMsfd = 4.739827684466089
bruteforceMsfd = 2.6249825991992437
gaMsfd = 2.4837635437946934

bestMicPositions = np.array([
    [ 0.75, 1.05, 1.2],
    [-0.85, 1.15, 1.2],
    [-0.25, 1.15, 1.2],
    [ 0.25, 1.15, 1.2],
])


# =========================================================
# Pipeline instance only to reuse computeMicGridPositions
# =========================================================

mesher = Mesher()
directSimulator = DirectSimulator()
evaluator = Evaluator()

pipeline = Pipeline(
    mesher=mesher,
    directSimulator=directSimulator,
    evaluator=evaluator,
)


# =========================================================
# Helpers
# =========================================================

def sortedVertices(vertices):
    keys = sorted(
        vertices.keys(),
        key=lambda key: int(key[1:])
    )

    return np.asarray(
        [vertices[key] for key in keys],
        dtype=float,
    )


def getPlotLimits(paramsList, margin=0.4):
    allPoints = []

    for params in paramsList:
        roomVertices = sortedVertices(params["data"]["vertices"])
        audienceVertices = sortedVertices(params["data"]["audience_area"])
        sourcePositions = np.asarray(params["data"]["source_pos"], dtype=float)

        allPoints.append(roomVertices)
        allPoints.append(audienceVertices)
        allPoints.append(sourcePositions[:, :2])
        allPoints.append(bestMicPositions[:, :2])

    allPoints = np.vstack(allPoints)

    xMin = np.min(allPoints[:, 0]) - margin
    xMax = np.max(allPoints[:, 0]) + margin
    yMin = np.min(allPoints[:, 1]) - margin
    yMax = np.max(allPoints[:, 1]) + margin

    return xMin, xMax, yMin, yMax


def plotRoomOnAxis(
    ax,
    params,
    possibleMicPositions,
    selectedMicPositions,
    title,
    xLimits,
    yLimits,
):
    roomVertices = sortedVertices(params["data"]["vertices"])
    audienceVertices = sortedVertices(params["data"]["audience_area"])

    roomPolygon = np.vstack([roomVertices, roomVertices[0]])
    audiencePolygon = np.vstack([audienceVertices, audienceVertices[0]])

    possibleMicPositions = np.asarray(possibleMicPositions, dtype=float)

    ax.plot(
        roomPolygon[:, 0],
        roomPolygon[:, 1],
        linewidth=2,
        label="Room",
    )

    ax.plot(
        audiencePolygon[:, 0],
        audiencePolygon[:, 1],
        linewidth=2,
        linestyle="--",
        label="Audience area",
    )

    ax.scatter(
        possibleMicPositions[:, 0],
        possibleMicPositions[:, 1],
        s=12,
        label="Possible mics",
    )

    sourcePositions = np.asarray(params["data"]["source_pos"], dtype=float)

    ax.scatter(
        sourcePositions[:, 0],
        sourcePositions[:, 1],
        s=80,
        marker="*",
        label="Sources",
    )

    for i, source in enumerate(sourcePositions):
        ax.text(
            source[0],
            source[1],
            f"S{i + 1}",
            fontsize=9,
        )

    selectedMicPositions = np.asarray(selectedMicPositions, dtype=float)

    ax.scatter(
        selectedMicPositions[:, 0],
        selectedMicPositions[:, 1],
        s=90,
        marker="x",
        label="Selected mics",
    )

    for i, mic in enumerate(selectedMicPositions):
        ax.text(
            mic[0],
            mic[1],
            f"M{i + 1}",
            fontsize=9,
        )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.set_xlim(xLimits)
    ax.set_ylim(yLimits)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)


# =========================================================
# Generate possible mic positions
# =========================================================

basePossibleMicPositions = pipeline.computeMicGridPositions(
    audienceArea=baseParams["data"]["audience_area"],
    micSpacing=0.1,
    micHeight=1.2,
    margin=0.0,
)

bruteforcePossibleMicPositions = pipeline.computeMicGridPositions(
    audienceArea=bruteforceParams["data"]["audience_area"],
    micSpacing=0.1,
    micHeight=1.2,
    margin=0.0,
)

gaPossibleMicPositions = pipeline.computeMicGridPositions(
    audienceArea=gaParams["data"]["audience_area"],
    micSpacing=0.1,
    micHeight=1.2,
    margin=0.0,
)


# =========================================================
# Plot figure
# =========================================================

paramsList = [
    baseParams,
    bruteforceParams,
    gaParams,
]

xMin, xMax, yMin, yMax = getPlotLimits(paramsList, margin=0.4)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6),
)

plotRoomOnAxis(
    ax=axes[0],
    params=baseParams,
    possibleMicPositions=basePossibleMicPositions,
    selectedMicPositions=bestMicPositions,
    title=f"Recinto base\nMSFD = {baseMsfd:.3f} dB",
    xLimits=(xMin, xMax),
    yLimits=(yMin, yMax),
)

plotRoomOnAxis(
    ax=axes[1],
    params=bruteforceParams,
    possibleMicPositions=bruteforcePossibleMicPositions,
    selectedMicPositions=bestMicPositions,
    title=f"Fuerza bruta\nMSFD = {bruteforceMsfd:.3f} dB",
    xLimits=(xMin, xMax),
    yLimits=(yMin, yMax),
)

plotRoomOnAxis(
    ax=axes[2],
    params=gaParams,
    possibleMicPositions=gaPossibleMicPositions,
    selectedMicPositions=bestMicPositions,
    title=f"Algoritmo genético\nMSFD = {gaMsfd:.3f} dB",
    xLimits=(xMin, xMax),
    yLimits=(yMin, yMax),
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
)

fig.suptitle(
    "Comparación de geometrías: recinto base, fuerza bruta y algoritmo genético",
    fontsize=14,
)

fig.tight_layout(rect=[0, 0.10, 1, 0.92])

outputPath = Path("data/comparison_plots/room_comparison_base_bruteforce_ga.png")
outputPath.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(outputPath, dpi=200)
plt.show()

print(f"Saved plot to: {outputPath}")