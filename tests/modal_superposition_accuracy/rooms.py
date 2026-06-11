import json
import copy
from pathlib import Path

import numpy as np


# =========================================================
# Rooms de prueba
# =========================================================

testRooms = {
    "room_04w_01": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [5.4, 0.2],
                "V3": [5.1, 3.6],
                "V4": [-0.2, 3.2],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
            },
            "Z": 3.0,
        }
    },

    "room_04w_02": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [6.0, 0.0],
                "V3": [5.4, 4.0],
                "V4": [0.3, 3.6],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
            },
            "Z": 3.2,
        }
    },

    "room_04w_03": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [4.8, -0.3],
                "V3": [5.5, 2.8],
                "V4": [0.2, 3.5],
            },
            "walls": {
                "W1": 3.0,
                "W2": -2.0,
                "W3": 4.0,
                "W4": -3.0,
            },
            "Z": 2.8,
        }
    },

    "room_04w_04": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [7.0, 0.4],
                "V3": [6.2, 3.2],
                "V4": [-0.4, 2.8],
            },
            "walls": {
                "W1": -4.0,
                "W2": 3.0,
                "W3": -2.5,
                "W4": 4.5,
            },
            "Z": 3.4,
        }
    },

    "room_05w_01": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [4.8, -0.2],
                "V3": [5.5, 2.0],
                "V4": [3.1, 4.0],
                "V5": [-0.3, 3.1],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
                "W5": 0.0,
            },
            "Z": 3.0,
        }
    },

    "room_05w_02": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [5.6, 0.1],
                "V3": [6.3, 2.7],
                "V4": [2.9, 4.3],
                "V5": [-0.6, 2.5],
            },
            "walls": {
                "W1": 2.5,
                "W2": -3.5,
                "W3": 4.0,
                "W4": -2.0,
                "W5": 3.0,
            },
            "Z": 3.3,
        }
    },

    "room_06w_01": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [3.0, -0.4],
                "V3": [6.0, 0.4],
                "V4": [6.4, 3.2],
                "V5": [3.1, 4.2],
                "V6": [-0.4, 2.8],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
                "W5": 0.0,
                "W6": 0.0,
            },
            "Z": 3.0,
        }
    },

    "room_06w_02": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [2.5, -0.5],
                "V3": [5.7, 0.1],
                "V4": [6.2, 2.9],
                "V5": [3.4, 4.5],
                "V6": [0.1, 3.4],
            },
            "walls": {
                "W1": -3.0,
                "W2": 2.0,
                "W3": -4.0,
                "W4": 3.5,
                "W5": -2.5,
                "W6": 4.0,
            },
            "Z": 3.5,
        }
    },

    "room_08w_01": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [2.0, -0.3],
                "V3": [4.5, 0.2],
                "V4": [5.6, 1.4],
                "V5": [5.0, 3.2],
                "V6": [3.0, 4.0],
                "V7": [0.7, 3.5],
                "V8": [-0.5, 1.6],
            },
            "walls": {
                "W1": 0.0,
                "W2": 0.0,
                "W3": 0.0,
                "W4": 0.0,
                "W5": 0.0,
                "W6": 0.0,
                "W7": 0.0,
                "W8": 0.0,
            },
            "Z": 3.0,
        }
    },

    "room_08w_02": {
        "data": {
            "vertices": {
                "V1": [0.0, 0.0],
                "V2": [1.8, -0.5],
                "V3": [4.2, -0.1],
                "V4": [6.1, 1.1],
                "V5": [5.8, 3.0],
                "V6": [4.0, 4.2],
                "V7": [1.3, 3.8],
                "V8": [-0.6, 1.9],
            },
            "walls": {
                "W1": 2.0,
                "W2": -3.0,
                "W3": 3.5,
                "W4": -2.5,
                "W5": 4.0,
                "W6": -3.5,
                "W7": 2.5,
                "W8": -2.0,
            },
            "Z": 3.4,
        }
    },
}


# =========================================================
# Geometría 2D auxiliar
# =========================================================

def getRoomVertices(roomParams):
    vertices = roomParams["data"]["vertices"]

    return np.asarray(
        [
            vertices[key]
            for key in sorted(vertices.keys(), key=lambda k: int(k[1:]))
        ],
        dtype=float
    )


def pointInPolygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

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


# =========================================================
# Generación de posiciones
# =========================================================

def generateRoomPositionConfigs(
    roomParams,
    nConfigs=5,
    nMics=4,
    seed=1234,
    sourceZ=1.4,
    micZ=1.2,
    wallMargin=0.55,
    minMicDistance=0.90,
    minSourceMicDistance=1.05,
):
    rng = np.random.default_rng(seed)
    polygon = getRoomVertices(roomParams)

    configs = {}

    for configIndex in range(1, nConfigs + 1):
        for _ in range(10000):
            sourceXY = sampleRoomPoint(
                rng=rng,
                polygon=polygon,
                wallMargin=wallMargin,
            )

            micsXY = []
            validConfig = True

            for _ in range(nMics):
                foundMic = False

                for _ in range(10000):
                    micXY = sampleRoomPoint(
                        rng=rng,
                        polygon=polygon,
                        wallMargin=wallMargin,
                    )

                    sourceMicDistance = np.linalg.norm(micXY - sourceXY)

                    if sourceMicDistance < minSourceMicDistance:
                        continue

                    tooCloseToOtherMic = any(
                        np.linalg.norm(micXY - otherMicXY) < minMicDistance
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
                configName = f"C{configIndex}"

                configs[configName] = {
                    "source": [
                        round(float(sourceXY[0]), 3),
                        round(float(sourceXY[1]), 3),
                        float(sourceZ),
                    ],
                    "mics": {
                        f"M{micIndex + 1}": [
                            round(float(micXY[0]), 3),
                            round(float(micXY[1]), 3),
                            float(micZ),
                        ]
                        for micIndex, micXY in enumerate(micsXY)
                    },
                }

                break

        else:
            raise RuntimeError("No se pudo generar una configuración válida.")

    return configs


def addPositionConfigsToRooms(
    testRooms,
    nConfigs=5,
    nMics=4,
    baseSeed=20260610,
):
    experimentRooms = copy.deepcopy(testRooms)

    for roomIndex, (roomName, roomParams) in enumerate(experimentRooms.items()):
        positionConfigs = generateRoomPositionConfigs(
            roomParams=roomParams,
            nConfigs=nConfigs,
            nMics=nMics,
            seed=baseSeed + roomIndex,
            sourceZ=1.4,
            micZ=1.2,
            wallMargin=0.55,
            minMicDistance=0.90,
            minSourceMicDistance=1.05,
        )

        roomParams["data"]["position_configs"] = positionConfigs

    return experimentRooms


# =========================================================
# Validación
# =========================================================

def validateExperimentRooms(
    experimentRooms,
    wallMargin=0.55,
    minMicDistance=0.90,
    minSourceMicDistance=1.05,
):
    for roomName, roomParams in experimentRooms.items():
        polygon = getRoomVertices(roomParams)
        configs = roomParams["data"]["position_configs"]

        for configName, config in configs.items():
            sourceXY = np.asarray(config["source"][:2], dtype=float)

            if not isValidRoomPoint(sourceXY, polygon, wallMargin):
                raise ValueError(f"{roomName} {configName}: fuente inválida")

            micsXY = []

            for micName, micPosition in config["mics"].items():
                micXY = np.asarray(micPosition[:2], dtype=float)

                if not isValidRoomPoint(micXY, polygon, wallMargin):
                    raise ValueError(f"{roomName} {configName}: {micName} inválido")

                if np.linalg.norm(micXY - sourceXY) < minSourceMicDistance:
                    raise ValueError(
                        f"{roomName} {configName}: {micName} muy cerca de la fuente"
                    )

                micsXY.append((micName, micXY))

            for i in range(len(micsXY)):
                micNameI, micXYI = micsXY[i]

                for j in range(i + 1, len(micsXY)):
                    micNameJ, micXYJ = micsXY[j]

                    if np.linalg.norm(micXYI - micXYJ) < minMicDistance:
                        raise ValueError(
                            f"{roomName} {configName}: {micNameI} y {micNameJ} muy cerca"
                        )

    print("Todas las salas y configuraciones son válidas.")


# =========================================================
# Guardado
# =========================================================

def saveExperimentRooms(experimentRooms, outputPath):
    outputPath = Path(outputPath)
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(experimentRooms, f, indent=4, ensure_ascii=False)

    print(f"JSON guardado en: {outputPath}")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    outputPath = "data/sm_accuracy_test/sm_accuracy_test_rooms.json"

    experimentRooms = addPositionConfigsToRooms(
        testRooms=testRooms,
        nConfigs=5,
        nMics=4,
        baseSeed=20260610,
    )

    validateExperimentRooms(
        experimentRooms,
        wallMargin=0.55,
        minMicDistance=0.90,
        minSourceMicDistance=1.05,
    )

    saveExperimentRooms(
        experimentRooms=experimentRooms,
        outputPath=outputPath,
    )