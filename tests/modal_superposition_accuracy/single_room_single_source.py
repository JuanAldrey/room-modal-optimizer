from pathlib import Path
import json


THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

INPUT_PATH = RESULTS_DIR / "single_square_many_configs.json"
OUTPUT_PATH = RESULTS_DIR / "single_square_fixed_source_many_mics.json"

# Opción A: tomar como fuente fija la source de C001
USE_FIRST_CONFIG_SOURCE = True

# Opción B: definirla manualmente
FIXED_SOURCE = [1.501, 1.083, 1.4]


def loadJson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def saveJson(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def convertToFixedSource(data):
    convertedData = {}

    for roomName, roomParams in data.items():
        roomData = roomParams["data"]
        positionConfigs = roomData["position_configs"]

        firstConfigName = next(iter(positionConfigs.keys()))

        if USE_FIRST_CONFIG_SOURCE:
            fixedSource = positionConfigs[firstConfigName]["source"]
        else:
            fixedSource = FIXED_SOURCE

        newPositionConfigs = {}

        for configName, config in positionConfigs.items():
            newPositionConfigs[configName] = {
                "mics": config["mics"]
            }

            if "description" in config:
                newPositionConfigs[configName]["description"] = config["description"]

        convertedData[roomName] = {
            "data": {
                "vertices": roomData["vertices"],
                "walls": roomData["walls"],
                "Z": roomData["Z"],
                "source": fixedSource,
                "position_configs": newPositionConfigs,
            }
        }

    return convertedData


def main():
    data = loadJson(INPUT_PATH)

    convertedData = convertToFixedSource(data)

    saveJson(convertedData, OUTPUT_PATH)

    print(f"JSON convertido guardado en: {OUTPUT_PATH}")

    for roomName, roomParams in convertedData.items():
        source = roomParams["data"]["source"]
        nConfigs = len(roomParams["data"]["position_configs"])

        print(f"Room: {roomName}")
        print(f"Fixed source: {source}")
        print(f"Mic configs: {nConfigs}")


if __name__ == "__main__":
    main()