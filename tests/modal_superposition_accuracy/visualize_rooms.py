import json
from pathlib import Path

from room_modal_optimizer.meshing.mesher import Mesher


# =========================================================
# Paths
# =========================================================

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

ROOMS_JSON = RESULTS_DIR / "sm_accuracy_test_rooms.json"


# =========================================================
# Helpers
# =========================================================

def loadJson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cleanRoomParams(roomParams):
    """
    El Mesher solo necesita vertices, walls y Z.
    Sacamos position_configs.
    """
    data = roomParams["data"]

    return {
        "data": {
            "vertices": data["vertices"],
            "walls": data["walls"],
            "Z": data["Z"],
        }
    }


def visualizeRooms():
    experimentRooms = loadJson(ROOMS_JSON)

    print(f"Rooms found: {len(experimentRooms)}")

    for roomIndex, (roomName, roomParams) in enumerate(experimentRooms.items(), start=1):
        print()
        print("=" * 80)
        print(f"[{roomIndex}/{len(experimentRooms)}] Visualizing {roomName}")
        print("=" * 80)

        cleanParams = cleanRoomParams(roomParams)

        visualRoomName = f"{roomName}_visual_check"

        mesher = Mesher()

        meshPath = mesher.create(
            cleanParams,
            room_name=visualRoomName,
            visualize=True,
        )

        print(f"Mesh generated: {meshPath}")

        input("Cerrá la ventana de Gmsh y apretá Enter para seguir con la próxima room...")


if __name__ == "__main__":
    visualizeRooms()