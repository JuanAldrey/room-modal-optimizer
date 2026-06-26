import numpy as np

class Evaluator:
    @staticmethod
    def pressure_to_db(response, eps=1e-12):
        """
        Convierte una respuesta compleja o lineal a magnitud en dB.

        Inputs:
            - response: array. Puede ser complejo o real.
            - eps: piso numérico para evitar log(0).

        Output:
            - response_db: array en dB.
        """

        response = np.asarray(response)

        return 20.0 * np.log10(np.abs(response) + eps)

    @staticmethod
    def magnitude_deviation(response_db):
        """
        Calcula la Magnitude Deviation (MD) según MSFD.

        Para cada receptor se calcula la desviación estándar de su respuesta
        en frecuencia. Luego se promedian esas desviaciones entre receptores.

        Inputs:
            - response_db: array [n_receivers, n_freqs] o [n_freqs].

        Output:
            - md: float [dB].
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
        Calcula la Spatial Deviation (SD) según MSFD.

        Para cada frecuencia se calcula la desviación estándar entre receptores,
        y luego se promedia en frecuencia.

        Inputs:
            - response_db: array [n_receivers, n_freqs] o [n_freqs].

        Output:
            - sd: float [dB].
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
        Calcula MSFD = wM * MD + wS * SD.

        Inputs:
            - response: respuesta fuente-receptor.
                Si input_is_db=False, puede ser compleja o lineal.
                Si input_is_db=True, se interpreta como dB.
                Forma esperada: [n_receivers, n_freqs].
            - input_is_db: bool.
            - weight_magnitude: peso de MD.
            - weight_spatial: peso de SD.
            - eps: piso numérico para conversión a dB.

        Outputs:
            - dict con MSFD, MD, SD y pesos usados.
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