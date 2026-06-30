import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class Pipeline:
    """
    Evaluates room geometries using direct FEM simulations and the MSFD metric.

    This pipeline receives a room configuration, generates its mesh, defines a grid
    of candidate microphone positions inside the audience area, simulates the
    frequency response from the source to all candidate receivers, and searches
    for the microphone combination that minimizes MSFD.

    The class is mainly used by geometry optimization routines, where each
    candidate room is evaluated by its best achievable microphone configuration.
    Failed geometries are stored in self.failedRooms.
    """
    def __init__(self, mesher, directSimulator, evaluator):
        self.mesher = mesher
        self.directSimulator = directSimulator
        self.evaluator = evaluator
        self.failedRooms = []

        self.savePlantPlot = False
        self.saveMicPlots = False

    def run(self, params, room_name='room', minMicDistance=0.5, nMics=4, fmax=200):
        """
        Runs the geometry evaluation pipeline for a single room configuration.

        The method generates the room mesh, creates a grid of candidate microphone
        positions inside the audience area, computes the direct FEM frequency response
        from the source to all candidate microphones, and searches for the microphone
        combination with the lowest MSFD value.

        Microphone combinations are generated randomly while enforcing a minimum
        distance between selected microphones. For each valid combination, the SPL
        responses are evaluated with the MSFD metric. The best MSFD value and the
        corresponding microphone positions are returned.

        If mesh generation fails, the room parameters are stored in self.failedRooms
        and the method returns None.

        Args:
            params (dict): Room parameters, including geometry data, source position
                and audience area under the "data" key.
            room_name (str): Name used to save mesh and optional plot outputs.
            minMicDistance (float): Minimum allowed distance between microphones in
                a selected combination, in meters.
            nMics (int): Number of microphones to select per evaluated combination.

        Returns:
            tuple[float, np.ndarray] | None: Best MSFD value and selected microphone
            positions with shape (nMics, 3). Returns None if mesh generation fails.
        """
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
        print("Amount of possible microphone positions:", possibleMicPositions.shape[0])

        if self.savePlantPlot:
            plotsDir = Path("data") / room_name / "plots/plants"

            plotsDir.mkdir(parents=True, exist_ok=True)

            self.plotMicLayout(
                params=params,
                possibleMicPositions=possibleMicPositions,
                selectedMicPositions=None,
                title="Possible mic positions",
                outputPath=plotsDir / "possible_mic_positions.png",
            )

        # Calculate transfer function from source to all microphones
        freqsOut, splResponses = self.directSimulator.simulate(
            mesh_path=meshPath,
            mic_positions=possibleMicPositions,
            order=1,
            room_name=room_name,
            freqs=np.arange(20.0, fmax, 2.0),
            use_impedance=True,
            wall_z=25.0 + 0j,
            floor_z=25.0 + 0j,
            ceiling_z=25.0 + 0j,
        )

        # Extract possible microphone combos and calculate best MSFD
        nCombos = int(
            (56 / 45) * possibleMicPositions.shape[0]**2
            - (80 / 3) * possibleMicPositions.shape[0]
            - 4000
        )
        nCombos = max(nCombos, 1000)
        nCombos = min(nCombos, 300000)
        combos = self.generateRandomCombos(
            possibleMicPositions=possibleMicPositions,
            nMicsPerCombo=nMics,
            nCombos=nCombos,
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


                # Plot for documenting
                if self.saveMicPlots:
                    self.plotMicLayout(
                        params=params,
                        possibleMicPositions=possibleMicPositions,
                        selectedMicPositions=bestMicPositions,
                        title=f"New best MSFD = {bestMsfd:.3f}",
                        outputPath=plotsDir / f"new_best_{len(str(bestMsfd))}_{bestMsfd:.3f}.png",
                    )

        bestMicPositions = possibleMicPositions[bestCombo]

        return bestMsfd, bestMicPositions

    def computeMicGridPositions(
        self,
        audienceArea,
        micSpacing=0.25,
        micHeight=1.2,
        margin=0.0,
    ):
        """
        Generates a regular grid of candidate microphone positions.

        The grid is created inside the bounding box of the audience area. A margin can
        be applied to shrink the valid region before placing microphones. Each
        microphone is placed at the center of a grid cell and assigned a constant
        height.

        Args:
            audienceArea (dict | list | np.ndarray): Audience area definition used to
                compute the valid receiver bounds.
            micSpacing (float): Distance between adjacent microphone candidates in
                meters.
            micHeight (float): Microphone height in meters.
            margin (float): Distance removed from each side of the audience bounds
                before generating the grid.

        Returns:
            np.ndarray: Candidate microphone positions with shape (n_positions, 3).

        Raises:
            ValueError: If the audience area becomes invalid after applying the margin,
            or if micSpacing is too large for the available area.
        """
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
        """
        Generates random valid microphone index combinations.

        The method randomly selects groups of microphone candidates from
        possibleMicPositions. Each combination is sorted, checked for duplicates, and
        accepted only if all selected microphones satisfy the minimum distance
        constraint.

        Args:
            possibleMicPositions (np.ndarray): Candidate microphone positions with
                shape (n_positions, 3).
            nMicsPerCombo (int): Number of microphones selected in each combination.
            nCombos (int): Target number of valid combinations to generate.
            minMicDistance (float): Minimum allowed distance between any pair of
                selected microphones, in meters.
            randomSeed (int): Seed used by the random number generator for
                reproducible combinations.

        Returns:
            list[np.ndarray]: List of valid microphone index combinations. Each array
            has shape (nMicsPerCombo,).
        """
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
        """
        Computes the rectangular bounds of the audience area.

        The audience area vertices are sorted by their numeric key order and converted
        to a NumPy array. The method then extracts the minimum and maximum x and y
        coordinates, which define the bounding box used to generate candidate
        microphone positions.

        Args:
            audienceArea (dict): Audience area vertices indexed by keys such as V1,
                V2, V3, with each value given as an [x, y] coordinate.

        Returns:
            tuple[float, float, float, float]: Bounds as (xMin, xMax, yMin, yMax).
        """
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
        """
        Checks whether all microphone pairs satisfy a minimum horizontal distance.

        The method computes pairwise distances between microphone positions using
        only the x and y coordinates. The z coordinate is ignored because the spacing
        constraint is applied in plan view.

        Args:
            micPositions (array-like): Microphone positions with shape (n_mics, 3).
            minDistance (float): Minimum allowed horizontal distance between any
                pair of microphones, in meters.

        Returns:
            bool: True if all microphone pairs satisfy the distance constraint,
            False otherwise.
        """
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
        """
        Plots the room layout, audience area, source positions and microphone positions.

        The method creates a 2D plan-view plot showing the room polygon, the audience
        area polygon, all candidate microphone positions and the source locations. If
        selectedMicPositions is provided, those microphones are highlighted and labeled
        separately.

        If outputPath is provided, the figure is saved to disk. Otherwise, the plot is
        displayed interactively.

        Args:
            params (dict): Room parameters containing room vertices, audience area and
                source positions under the "data" key.
            possibleMicPositions (array-like): Candidate microphone positions with
                shape (n_positions, 3).
            selectedMicPositions (array-like | None): Selected microphone positions
                with shape (n_selected, 3). If None, no selected microphones are shown.
            title (str): Plot title.
            outputPath (str | Path | None): Output image path. If None, the plot is
                shown instead of saved.

        Returns:
            None
        """
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

        sourcePositions = np.asarray(params["data"]["source_pos"], dtype=float)

        plt.scatter(
            sourcePositions[:, 0],
            sourcePositions[:, 1],
            s=80,
            marker="*",
            label="Sources",
        )

        for i, source in enumerate(sourcePositions):
            plt.text(source[0], source[1], f"S{i + 1}", fontsize=9)

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
        """
        Returns vertices sorted by their numeric key order.

        The input dictionary is expected to use keys such as V1, V2 and V3. The
        method sorts those keys numerically and returns the corresponding coordinates
        as a NumPy array.

        Args:
            vertices (dict): Dictionary of vertices, where each key is a vertex name
                and each value is an [x, y] coordinate.

        Returns:
            np.ndarray: Sorted vertex coordinates with shape (n_vertices, 2).
        """
        keys = sorted(
            vertices.keys(),
            key=lambda key: int(key[1:])
        )

        return np.asarray(
            [vertices[key] for key in keys],
            dtype=float,
        )
