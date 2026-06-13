import numpy as np


class ModalEvaluator:

    @staticmethod
    def evaluate(f_modes, n):
        """
        Frequency Spacing Index según Rindel.

        Inputs:
            - f_modes: array type object. Modes frequencies vector.
            - n: int type object. Max mode to evaluate.
        """

        f = np.asarray(f_modes, dtype=float)[:n]

        if len(f) < 2:
            raise ValueError("Se necesitan al menos 2 modos para calcular el FSI.")

        avg_frq_sp = (f[-1] - f[0]) / (n - 1)
        dif_nhb_modes = np.diff(f)
        sum_arg = (dif_nhb_modes / avg_frq_sp) ** 2
        frq_sp_idx = (1 / (n - 1)) * np.sum(sum_arg, axis=0)

        return float(frq_sp_idx)

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
        Calcula la desviación en magnitud (MD).

        Se toma la respuesta promedio entre receptores y se mide su
        desviación respecto de su propio nivel medio. De esta forma,
        la métrica penaliza la ondulación espectral, no el nivel absoluto.

        Inputs:
            - response_db: array [n_receivers, n_freqs] o [n_freqs].

        Output:
            - md: float.
        """

        response_db = np.asarray(response_db, dtype=float)

        if response_db.ndim == 1:
            mean_response = response_db
        elif response_db.ndim == 2:
            mean_response = np.mean(response_db, axis=0)
        else:
            raise ValueError(
                "response_db debe tener forma [n_freqs] o [n_receivers, n_freqs]."
            )

        mean_level = np.mean(mean_response)

        md = np.mean(np.abs(mean_response - mean_level))

        return float(md)

    @staticmethod
    def spatial_deviation(response_db):
        """
        Calcula la desviación espacial (SD).

        Para cada frecuencia se calcula el desvío estándar entre receptores,
        y luego se promedia en frecuencia.

        Inputs:
            - response_db: array [n_receivers, n_freqs] o [n_freqs].

        Output:
            - sd: float.
        """

        response_db = np.asarray(response_db, dtype=float)

        if response_db.ndim == 1:
            return 0.0

        if response_db.ndim != 2:
            raise ValueError(
                "response_db debe tener forma [n_freqs] o [n_receivers, n_freqs]."
            )

        sd_by_freq = np.std(response_db, axis=0)

        sd = np.mean(sd_by_freq)

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
            response_db = ModalEvaluator.pressure_to_db(response, eps=eps)

        md = ModalEvaluator.magnitude_deviation(response_db)
        sd = ModalEvaluator.spatial_deviation(response_db)

        msfd = weight_magnitude * md + weight_spatial * sd

        return {
            "MSFD": float(msfd),
            "MD": float(md),
            "SD": float(sd),
            "weight_magnitude": float(weight_magnitude),
            "weight_spatial": float(weight_spatial),
        }

    @staticmethod
    def evaluate_geometry_merit(
        f_modes,
        response,
        n_modes=25,
        input_is_db=False,
        weight_rindel=1.0,
        weight_msfd=1.0,
        weight_magnitude=0.5,
        weight_spatial=0.5,
        eps=1e-12,
    ):
        """
        Figura de mérito combinada para evaluación geométrica.

        Combina:
            - Penalización de Rindel: abs(FSI - 1)
            - MSFD calculado sobre respuesta modal aproximada

        Ojo: en una etapa posterior conviene normalizar ambas magnitudes
        antes de combinarlas, porque no necesariamente tienen la misma escala.
        """

        fsi = ModalEvaluator.evaluate(f_modes, n_modes)
        rindel_penalty = abs(fsi - 1.0)

        msfd_result = ModalEvaluator.evaluate_msfd(
            response=response,
            input_is_db=input_is_db,
            weight_magnitude=weight_magnitude,
            weight_spatial=weight_spatial,
            eps=eps,
        )

        merit = (
            weight_rindel * rindel_penalty
            + weight_msfd * msfd_result["MSFD"]
        )

        return {
            "merit": float(merit),
            "FSI": float(fsi),
            "rindel_penalty": float(rindel_penalty),
            "MSFD": msfd_result["MSFD"],
            "MD": msfd_result["MD"],
            "SD": msfd_result["SD"],
            "weight_rindel": float(weight_rindel),
            "weight_msfd": float(weight_msfd),
            "weight_magnitude": float(weight_magnitude),
            "weight_spatial": float(weight_spatial),
        }