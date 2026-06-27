class AbsorptionPipeline:
    """
    Evaluates absorber configurations on a patched room mesh.

    This pipeline receives an already generated patched mesh, assigns impedance
    values to the available surface patches, runs a direct FEM simulation, and
    evaluates the resulting SPL responses using the MSFD metric.

    It is mainly intended for absorber optimization, where the room geometry is
    fixed and the optimization variables are the impedance or resonator types
    assigned to each patch.
    """
    def __init__(self, directSimulator, evaluator):
        self.directSimulator = directSimulator
        self.evaluator = evaluator
    
    def run(self, mesh_path, impedance_mappings, mic_positions, room_name="room"):
        """
        Runs the absorber evaluation for a given impedance mapping.

        The method simulates the patched room mesh using the provided impedance
        mapping, evaluates the resulting SPL responses at the microphone positions,
        and returns the MSFD value.

        Args:
            mesh_path (str | Path): Path to the patched Gmsh mesh file.
            impedance_mappings (dict): Mapping between patch physical tags and
                resonator or impedance identifiers.
            mic_positions (array-like): Microphone positions with shape
                (n_mics, 3).
            room_name (str): Name used to identify the current room simulation.

        Returns:
            float: MSFD value for the evaluated absorber configuration.
        """
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

