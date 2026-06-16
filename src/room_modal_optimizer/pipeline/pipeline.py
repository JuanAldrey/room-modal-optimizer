import numpy as np

class Pipeline:
    def __init__(self, mesher, modalSimulator, directSimulator,evaluator):
        self.mesher = mesher
        self.modalSimulator = modalSimulator
        self.directSimulator = directSimulator
        self.evaluator = evaluator
        self.failedRooms = []

    def run(self, params, source_pos, room_name='room'):
        # Generate mesh
        meshPath = self.mesher.create(params, lc=0.2, source_pos=source_pos, room_name=room_name)
        if meshPath is None:
            self.failedRooms.append(params)
            return None
        
        # Calculate modes and source weights
        modal_result = self.modalSimulator.simulate(
            mesh_path=meshPath,
            order=2,
            room_name=room_name,
            target_freq=100.0,
            n_modes=150,
            tol=1e-8,
        )

        sourceWeights = modal_result["source_weights"]

        # Define possible microphone positions base on audience area
        possibleMicPositions = self.computePossibleMicPositions(params['data']['audience_area'])

        # Find best microphone positions configuration
        bestModalMsfd = None
        bestMicPositions = None

        for micPositions in possibleMicPositions:
            HModal = self.modalSimulator.modalTransferFromFixedSurfaceSource(
                receiverPositions=micPositions,
                freqs=np.arange(20.0, 201.0, 2.0),
                sourceWeights=sourceWeights,
                zeta=0.015,
                sourceStrength=0.01,
            )

            modalMsfd = self.evaluator.evaluate_msfd(
                response=HModal,
                input_is_db=False,
                weight_magnitude=0.5,
                weight_spatial=0.5,
            )["MSFD"]

            if bestModalMsfd is None or modalMsfd < bestModalMsfd:
                bestModalMsfd = modalMsfd
                bestMicPositions = micPositions

        # Calculate response with direct simulator
        freqsOut, splResponses = self.directSimulator.simulate(
            mesh_path=meshPath,
            mic_positions=bestMicPositions,
            room_name=room_name,
            freqs=np.arange(20.0, 201.0, 2.0),
            use_impedance=True,
            wall_z=25.0 + 0j,
            floor_z=25.0 + 0j,
            ceiling_z=25.0 + 0j,
        )

        # Return definitive MSFD
        return self.evaluator.evaluate_msfd(
                response=splResponses,
                input_is_db=True,
                weight_magnitude=0.5,
                weight_spatial=0.5,
            )["MSFD"]


    def computePossibleMicPositions(
        self,
        audienceArea,
        nMics=5,
        micHeight=1.2,
        minDistance=0.5,
        gridStep=0.25,
        nConfigs=200,
        randomSeed=1234,
    ):
        polygon = self.parseAudienceArea(audienceArea)

        candidatePoints = self.generateCandidateMicGrid(
            polygon=polygon,
            micHeight=micHeight,
            gridStep=gridStep,
        )

        if len(candidatePoints) < nMics:
            raise ValueError(
                f"No hay suficientes puntos candidatos: {len(candidatePoints)} "
                f"para nMics={nMics}"
            )

        rng = np.random.default_rng(randomSeed)

        configs = []
        tries = 0
        maxTries = nConfigs * 200

        while len(configs) < nConfigs and tries < maxTries:
            tries += 1

            indices = rng.choice(
                len(candidatePoints),
                size=nMics,
                replace=False,
            )

            micPositions = candidatePoints[indices]

            if self.hasMinimumDistance(micPositions, minDistance):
                configs.append([
                    tuple(mic)
                    for mic in micPositions
                ])

        if len(configs) == 0:
            raise ValueError("No se pudo generar ninguna configuración válida de mics.")

        return configs
    
    def parseAudienceArea(self, audienceArea):
        if isinstance(audienceArea, dict):
            vertices = audienceArea["vertices"]

            if isinstance(vertices, dict):
                keys = sorted(
                    vertices.keys(),
                    key=lambda key: int(key[1:])
                )

                polygon = [
                    vertices[key]
                    for key in keys
                ]

            else:
                polygon = vertices

        else:
            polygon = audienceArea

        polygon = np.asarray(polygon, dtype=float)

        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError(f"audienceArea inválida: shape={polygon.shape}")

        return polygon


    def generateCandidateMicGrid(self, polygon, micHeight, gridStep):
        minX = np.min(polygon[:, 0])
        maxX = np.max(polygon[:, 0])
        minY = np.min(polygon[:, 1])
        maxY = np.max(polygon[:, 1])

        xs = np.arange(minX, maxX + 0.5 * gridStep, gridStep)
        ys = np.arange(minY, maxY + 0.5 * gridStep, gridStep)

        points = []

        for x in xs:
            for y in ys:
                if self.pointInPolygon((x, y), polygon):
                    points.append([x, y, micHeight])

        return np.asarray(points, dtype=float)


    def pointInPolygon(self, point, polygon):
        x, y = point
        inside = False

        n = len(polygon)
        j = n - 1

        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            intersects = (
                ((yi > y) != (yj > y))
                and (
                    x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-15) + xi
                )
            )

            if intersects:
                inside = not inside

            j = i

        return inside

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