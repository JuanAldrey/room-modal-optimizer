import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class Pipeline:
    def __init__(self, mesher, directSimulator, evaluator):
        self.mesher = mesher
        self.directSimulator = directSimulator
        self.evaluator = evaluator
        self.failedRooms = []

    def run(self, params, room_name='room', minMicDistance=0.5):
        # Generate mesh
        meshPath = self.mesher.create(params, lc=0.28, source_pos=params["data"]["source_pos"], room_name=room_name)
        if meshPath is None:
            self.failedRooms.append(params)
            return None

        # Define possible microphone positions based on audience area
        print("Defining possible microphone positions...")
        possibleMicPositions = self.computeMicGridPositions(
            audienceArea=params["data"]["audience_area"],
            micSpacing=0.1,
            micHeight=1.2,
            margin=0.0,
        )
        print("Amount of possible microphone positions:", possibleMicPositions.shape)

        plotsDir = Path("data") / room_name / "plots"
        plotsDir.mkdir(parents=True, exist_ok=True)

        """
        # Plots for documenting
        self.plotMicLayout(
            params=params,
            possibleMicPositions=possibleMicPositions,
            selectedMicPositions=None,
            title="Possible mic positions",
            outputPath=plotsDir / "possible_mic_positions.png",
        )
        """

        # Calculate transfer function from source to all microphones
        freqsOut, splResponses = self.directSimulator.simulate(
            mesh_path=meshPath,
            mic_positions=possibleMicPositions,
            order=1,
            room_name=room_name,
            freqs=np.arange(20.0, 201.0, 2.0),
            use_impedance=True,
            wall_z=25.0 + 0j,
            floor_z=25.0 + 0j,
            ceiling_z=25.0 + 0j,
        )

        # Extract possible microphone combos and calculate best MSFD
        combos = self.generateRandomCombos(
            possibleMicPositions=possibleMicPositions,
            nMicsPerCombo=4,
            nCombos=100000,
            minMicDistance=minMicDistance,
        )

        print("Amount of combos: ", len(combos))

        bestMsfd = np.inf
        bestCombo = None

        for combo in combos:
            response = splResponses[combo, :]

            msfd = self.evaluator.evaluate_msfd(
                response=response,
                input_is_db=True,
                weight_magnitude=0.5,
                weight_spatial=0.5,
            )["MSFD"]

            if msfd < bestMsfd:
                bestMsfd = msfd
                bestCombo = combo
                print("New best: ", bestMsfd, " with mics ", bestCombo)

                bestMicPositions = possibleMicPositions[bestCombo]

                """
                # Plot for documenting
                self.plotMicLayout(
                    params=params,
                    possibleMicPositions=possibleMicPositions,
                    selectedMicPositions=bestMicPositions,
                    title=f"New best MSFD = {bestMsfd:.3f}",
                    outputPath=plotsDir / f"new_best_{len(str(bestMsfd))}_{bestMsfd:.3f}.png",
                )
                """

        bestMicPositions = possibleMicPositions[bestCombo]

        return bestMsfd, bestMicPositions

    def computeMicGridPositions(
        self,
        audienceArea,
        micSpacing=0.25,
        micHeight=1.2,
        margin=0.0,
    ):
        xMin, xMax, yMin, yMax = self.getAudienceBounds(audienceArea)

        xMin += margin
        xMax -= margin
        yMin += margin
        yMax -= margin

        if xMin >= xMax or yMin >= yMax:
            raise ValueError("El audience_area queda inválida después de aplicar margin.")

        width = xMax - xMin
        depth = yMax - yMin

        nX = int(np.floor(width / micSpacing))
        nY = int(np.floor(depth / micSpacing))

        if nX <= 0 or nY <= 0:
            raise ValueError("El micSpacing es demasiado grande para el audience_area.")

        xs = xMin + micSpacing * (0.5 + np.arange(nX))
        ys = yMin + micSpacing * (0.5 + np.arange(nY))

        micPositions = []

        for y in ys:
            for x in xs:
                micPositions.append((float(x), float(y), float(micHeight)))

        return np.asarray(micPositions, dtype=float)
    
    def generateRandomCombos(
        self,
        possibleMicPositions,
        nMicsPerCombo=4,
        nCombos=100,
        minMicDistance=0.5,
        randomSeed=1234,
    ):
        rng = np.random.default_rng(randomSeed)

        combos = []
        seen = set()
        maxTries = nCombos * 500

        for _ in range(maxTries):
            if len(combos) >= nCombos:
                break

            combo = rng.choice(
                len(possibleMicPositions),
                size=nMicsPerCombo,
                replace=False,
            )

            combo = tuple(sorted(int(i) for i in combo))

            if combo in seen:
                continue

            micPositions = possibleMicPositions[list(combo)]

            if not self.hasMinimumDistance(micPositions, minMicDistance):
                continue

            seen.add(combo)
            combos.append(np.asarray(combo, dtype=int))

        return combos
    
    def getAudienceBounds(self, audienceArea):
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
    
    def hasMinimumDistance(self, micPositions, minDistance):
        micPositions = np.asarray(micPositions, dtype=float)

        for i in range(len(micPositions)):
            for j in range(i + 1, len(micPositions)):
                distance = np.linalg.norm(
                    micPositions[i, :2] - micPositions[j, :2]
                )

                if distance < minDistance:
                    return False

        return True


    # Plotting function for documenting
    def plotMicLayout(
        self,
        params,
        possibleMicPositions,
        selectedMicPositions=None,
        title="Mic layout",
        outputPath=None,
    ):
        roomVertices = self.sortedVertices(params["data"]["vertices"])
        audienceVertices = self.sortedVertices(params["data"]["audience_area"])

        roomPolygon = np.vstack([roomVertices, roomVertices[0]])
        audiencePolygon = np.vstack([audienceVertices, audienceVertices[0]])

        possibleMicPositions = np.asarray(possibleMicPositions, dtype=float)

        plt.figure(figsize=(7, 7))

        plt.plot(roomPolygon[:, 0], roomPolygon[:, 1], linewidth=2, label="Room")
        plt.plot(audiencePolygon[:, 0], audiencePolygon[:, 1], linewidth=2, linestyle="--", label="Audience area")

        plt.scatter(
            possibleMicPositions[:, 0],
            possibleMicPositions[:, 1],
            s=20,
            label="Possible mics",
        )

        sourcePos = np.asarray(params["data"]["source_pos"], dtype=float)

        plt.scatter(
            sourcePos[0],
            sourcePos[1],
            s=80,
            marker="*",
            label="Source",
        )

        if selectedMicPositions is not None:
            selectedMicPositions = np.asarray(selectedMicPositions, dtype=float)

            plt.scatter(
                selectedMicPositions[:, 0],
                selectedMicPositions[:, 1],
                s=100,
                marker="x",
                label="Selected mics",
            )

            for i, mic in enumerate(selectedMicPositions):
                plt.text(mic[0], mic[1], f"M{i + 1}", fontsize=9)

        plt.title(title)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if outputPath is not None:
            outputPath = Path(outputPath)
            outputPath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(outputPath, dpi=150)
            plt.close()
        else:
            plt.show()

    def sortedVertices(self, vertices):
        keys = sorted(
            vertices.keys(),
            key=lambda key: int(key[1:])
        )

        return np.asarray(
            [vertices[key] for key in keys],
            dtype=float,
        )
