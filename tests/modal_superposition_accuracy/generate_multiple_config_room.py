import json
from pathlib import Path

import numpy as np


OUTPUT_PATH = Path(
    "tests/modal_superposition_accuracy/results/single_square_many_configs.json"
)

N_CONFIGS = 50
N_MICS = 4
SEED = 20260612

ROOM_NAME = "single_square_5x5x3"

SOURCE_Z = 1.4
MIC_Z = 1.2

WALL_MARGIN = 0.60
MIN_MIC_DISTANCE = 0.90
MIN_SOURCE_MIC_DISTANCE = 1.10


testRooms = {
    ROOM_NAME: {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [5.0, 0.0],
                "V3": [5.0, 3.0],
                "V4": [0.0, 3.0],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
            },
            "Z": 3.0,
        }
    }
}


def getRoomVertices(roomParams):
    vertices = roomParams["data"]["vertices"]

    return np.asarray(
        [
            vertices[key]
            for key in sorted(vertices.keys(), key=lambda k: int(k[1:]))
        ],
        dtype=float,
    )


def pointInPolygon(point, polygon):
    x, y = point
    inside = False

    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]

        crosses = (y1 > y) != (y2 > y)

        if crosses:
            xIntersection = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1

            if x < xIntersection:
                inside = not inside

    return inside


def distancePointToSegment(point, a, b):
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ab = b - a
    abNorm2 = np.dot(ab, ab)

    if abNorm2 == 0:
        return np.linalg.norm(point - a)

    t = np.dot(point - a, ab) / abNorm2
    t = np.clip(t, 0.0, 1.0)

    closest = a + t * ab

    return np.linalg.norm(point - closest)


def minDistanceToWalls(point, polygon):
    distances = []

    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]

        distances.append(distancePointToSegment(point, a, b))

    return min(distances)


def isValidRoomPoint(point, polygon, wallMargin):
    if not pointInPolygon(point, polygon):
        return False

    if minDistanceToWalls(point, polygon) < wallMargin:
        return False

    return True


def sampleRoomPoint(rng, polygon, wallMargin, maxTries=10000):
    minXY = polygon.min(axis=0)
    maxXY = polygon.max(axis=0)

    for _ in range(maxTries):
        point = rng.uniform(minXY, maxXY)

        if isValidRoomPoint(point, polygon, wallMargin):
            return point

    raise RuntimeError("No se pudo generar un punto válido dentro del recinto.")


def generatePositionConfigs(roomParams):
    rng = np.random.default_rng(SEED)
    polygon = getRoomVertices(roomParams)

    configs = {}

    for configIndex in range(1, N_CONFIGS + 1):
        for _ in range(10000):
            sourceXY = sampleRoomPoint(
                rng=rng,
                polygon=polygon,
                wallMargin=WALL_MARGIN,
            )

            micsXY = []
            validConfig = True

            for _ in range(N_MICS):
                foundMic = False

                for _ in range(10000):
                    micXY = sampleRoomPoint(
                        rng=rng,
                        polygon=polygon,
                        wallMargin=WALL_MARGIN,
                    )

                    if np.linalg.norm(micXY - sourceXY) < MIN_SOURCE_MIC_DISTANCE:
                        continue

                    tooCloseToOtherMic = any(
                        np.linalg.norm(micXY - otherMicXY) < MIN_MIC_DISTANCE
                        for otherMicXY in micsXY
                    )

                    if tooCloseToOtherMic:
                        continue

                    micsXY.append(micXY)
                    foundMic = True
                    break

                if not foundMic:
                    validConfig = False
                    break

            if validConfig:
                configName = f"C{configIndex:03d}"

                configs[configName] = {
                    "source": [
                        round(float(sourceXY[0]), 3),
                        round(float(sourceXY[1]), 3),
                        float(SOURCE_Z),
                    ],
                    "mics": {
                        f"M{micIndex + 1}": [
                            round(float(micXY[0]), 3),
                            round(float(micXY[1]), 3),
                            float(MIC_Z),
                        ]
                        for micIndex, micXY in enumerate(micsXY)
                    },
                }

                break

        else:
            raise RuntimeError(f"No se pudo generar la config {configIndex}")

    return configs


def validateConfigs(roomParams):
    polygon = getRoomVertices(roomParams)
    configs = roomParams["data"]["position_configs"]

    for configName, config in configs.items():
        sourceXY = np.asarray(config["source"][:2], dtype=float)

        if not isValidRoomPoint(sourceXY, polygon, WALL_MARGIN):
            raise ValueError(f"{configName}: fuente inválida")

        micsXY = []

        for micName, micPosition in config["mics"].items():
            micXY = np.asarray(micPosition[:2], dtype=float)

            if not isValidRoomPoint(micXY, polygon, WALL_MARGIN):
                raise ValueError(f"{configName}: {micName} inválido")

            if np.linalg.norm(micXY - sourceXY) < MIN_SOURCE_MIC_DISTANCE:
                raise ValueError(f"{configName}: {micName} muy cerca de la fuente")

            micsXY.append((micName, micXY))

        for i in range(len(micsXY)):
            micNameI, micXYI = micsXY[i]

            for j in range(i + 1, len(micsXY)):
                micNameJ, micXYJ = micsXY[j]

                if np.linalg.norm(micXYI - micXYJ) < MIN_MIC_DISTANCE:
                    raise ValueError(
                        f"{configName}: {micNameI} y {micNameJ} muy cerca"
                    )


def main():
    roomParams = testRooms[ROOM_NAME]

    roomParams["data"]["position_configs"] = generatePositionConfigs(roomParams)

    validateConfigs(roomParams)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(testRooms, f, indent=4, ensure_ascii=False)

    print(f"JSON guardado en: {OUTPUT_PATH}")
    print(f"Room: {ROOM_NAME}")
    print(f"Configs: {N_CONFIGS}")
    print("Validación OK.")


if __name__ == "__main__":
    main()