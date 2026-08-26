from scipy.stats import ks_2samp


class DriftDetector:

    @staticmethod
    def detect(
        reference,
        current,
        alpha=0.05,
    ):

        statistic, p = ks_2samp(
            reference,
            current
        )

        return {

            "statistic": statistic,

            "p_value": p,

            "drift": p < alpha
        }