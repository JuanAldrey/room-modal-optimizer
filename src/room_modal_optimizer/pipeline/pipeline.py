import numpy as np

class Pipeline:
    def __init__(self, mesher, modalSimulator, directSimulator, evaluator):
        self.mesher = mesher
        self.modalSimulator = modalSimulator
        self.directSimulator = directSimulator
        self.evaluator = evaluator
        self.failedRooms = []

    def run(self, params, room_name='room'):
        # Generate mesh
        meshPath = self.mesher.create(params, lc=0.28, source_pos=params["data"]["source_pos"], room_name=room_name)
        if meshPath is None:
            self.failedRooms.append(params)
            return None
        
        # Calculate modes and source weights
        print("Calculating modes...")
        modal_result = self.modalSimulator.simulate(
            mesh_path=meshPath,
            order=2,
            room_name=room_name,
            target_freq=100.0,
            n_modes=150,
            tol=1e-8,
        )

        sourceWeights = modal_result["source_weights"]

        # Define possible microphone positions based on audience area
        print("Defining possible microphone positions...")
        possibleMicPositions = self.computePossibleMicPositions(params['data']['audience_area'])
        print("Possible microphone positions: ", len(possibleMicPositions))

        # Find best microphone positions configuration
        print("Evaluating possible microphone positions...")
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
                print("New best: ", modalMsfd)
                bestModalMsfd = modalMsfd
                bestMicPositions = micPositions

        # Calculate response with direct simulator
        print("Calculating direct response")
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
        print("Calculating final MSFD")
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
        nConfigs=200,
        randomSeed=1234,
        margin=0.1,
    ):
        xMin, xMax, yMin, yMax = self.getAudienceBounds(audienceArea)

        xMin += margin
        xMax -= margin
        yMin += margin
        yMax -= margin

        if xMin >= xMax or yMin >= yMax:
            raise ValueError("El audience_area queda inválida después de aplicar margin.")

        rng = np.random.default_rng(randomSeed)

        configs = []
        maxTries = nConfigs * 200

        for _ in range(maxTries):
            if len(configs) >= nConfigs:
                break

            xs = rng.uniform(xMin, xMax, size=nMics)
            ys = rng.uniform(yMin, yMax, size=nMics)
            zs = np.full(nMics, micHeight)

            micPositions = np.column_stack([xs, ys, zs])

            if self.hasMinimumDistance(micPositions, minDistance):
                configs.append([
                    tuple(mic)
                    for mic in micPositions
                ])

        if len(configs) == 0:
            raise ValueError("No se pudo generar ninguna configuración válida de mics.")

        return configs
    
    def getAudienceBounds(self, audienceArea):
        vertices = audienceArea

        keys = sorted(
            vertices.keys(),
            key=lambda key: int(key[1:])
        )

        points = np.asarray(
            [vertices[key] for key in keys],
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