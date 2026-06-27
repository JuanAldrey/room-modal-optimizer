import numpy as np

class Evaluator:
    """
    Computes acoustic objective metrics from simulated frequency responses.

    This class provides static helper methods to convert linear or complex
    responses to dB and to compute the MSFD metric, including its Magnitude
    Deviation (MD) and Spatial Deviation (SD) components.
    """
    @staticmethod
    def pressure_to_db(response, eps=1e-12):
        """
        Converts a complex or linear response magnitude to dB.

        Args:
            response (array-like): Complex or real-valued response.
            eps (float): Numerical floor used to avoid log(0).

        Returns:
            np.ndarray: Response magnitude in dB.
        """

        response = np.asarray(response)

        return 20.0 * np.log10(np.abs(response) + eps)

    @staticmethod
    def magnitude_deviation(response_db):
        """
        Computes the Magnitude Deviation (MD) component of MSFD.

        For each receiver, the method computes the standard deviation of the
        frequency response. The final MD value is obtained by averaging those
        deviations across receivers.

        Args:
            response_db (array-like): Response in dB with shape
                (n_receivers, n_freqs) or (n_freqs,).

        Returns:
            float: Magnitude Deviation in dB.

        Raises:
            ValueError: If response_db is not a 1D or 2D array.
        """

        response_db = np.asarray(response_db, dtype=float)

        if response_db.ndim == 1:
            return float(np.std(response_db, ddof=1))

        if response_db.ndim != 2:
            raise ValueError(
                "response_db debe tener forma [n_freqs] o [n_receivers, n_freqs]."
            )

        mdByReceiver = np.std(response_db, axis=1, ddof=1)

        return float(np.mean(mdByReceiver))

    @staticmethod
    def spatial_deviation(response_db):
        """
        Computes the Spatial Deviation (SD) component of MSFD.

        For each frequency, the method computes the standard deviation across
        receivers. The final SD value is obtained by averaging those deviations
        across frequency.

        Args:
            response_db (array-like): Response in dB with shape
                (n_receivers, n_freqs) or (n_freqs,).

        Returns:
            float: Spatial Deviation in dB. Returns 0.0 when only one receiver
            is available.

        Raises:
            ValueError: If response_db is not a 1D or 2D array.
        """

        response_db = np.asarray(response_db, dtype=float)

        if response_db.ndim == 1:
            return 0.0

        if response_db.ndim != 2:
            raise ValueError(
                "response_db debe tener forma [n_receivers, n_freqs]."
            )

        nReceivers = response_db.shape[0]

        if nReceivers < 2:
            return 0.0

        sdByFreq = np.std(response_db, axis=0, ddof=1)

        sd = np.mean(sdByFreq)

        return float(sd)

    @staticmethod
    def evaluate_msfd(
        response,
        input_is_db=False,
        weight_magnitude=0.5,
        weight_spatial=0.5,
        eps=1e-12,
    ):
        """
        Computes the Mean Spatial Frequency Deviation (MSFD).

        MSFD is computed as a weighted sum of Magnitude Deviation and Spatial
        Deviation:

            MSFD = weight_magnitude * MD + weight_spatial * SD

        If input_is_db is False, the input response is first converted to dB using
        pressure_to_db(). If input_is_db is True, the input is assumed to already
        be expressed in dB.

        Args:
            response (array-like): Source-receiver response. Expected shape is
                (n_receivers, n_freqs), although a single receiver response with
                shape (n_freqs,) is also accepted.
            input_is_db (bool): If True, response is interpreted as dB. If False,
                response is interpreted as a linear or complex response.
            weight_magnitude (float): Weight applied to the MD component.
            weight_spatial (float): Weight applied to the SD component.
            eps (float): Numerical floor used when converting to dB.

        Returns:
            dict: Dictionary containing MSFD, MD, SD and the weights used.
        """

        if input_is_db:
            response_db = np.asarray(response, dtype=float)
        else:
            response_db = Evaluator.pressure_to_db(response, eps=eps)

        md = Evaluator.magnitude_deviation(response_db)
        sd = Evaluator.spatial_deviation(response_db)

        msfd = weight_magnitude * md + weight_spatial * sd

        return {
            "MSFD": float(msfd),
            "MD": float(md),
            "SD": float(sd),
            "weight_magnitude": float(weight_magnitude),
            "weight_spatial": float(weight_spatial),
        }