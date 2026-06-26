class AbsorptionPipeline:
    def __init__(self, directSimulator, evaluator):
        self.directSimulator = directSimulator
        self.evaluator = evaluator
    
    def run(self, mesh_path, impedance_mappings, mic_positions, room_name="room"):
        freqs, spl_responses = self.directSimulator.simulate(
            mesh_path,
            mic_positions=mic_positions,
            room_name=room_name,
            patch=True,
            impedance_mappings=impedance_mappings
        )

        return self.evaluator.evaluate_msfd(
                response=spl_responses,
                input_is_db=True,
                weight_magnitude=0.5,
                weight_spatial=0.5,
            )["MSFD"]

