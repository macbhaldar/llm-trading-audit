import numpy as np

class Calibration:

    @staticmethod
    def expected_calibration_error(
        confidence,
        correctness,
        bins=10,
    ):

        confidence = np.asarray(confidence)
        correctness = np.asarray(correctness)
        edges = np.linspace(0, 1, bins + 1)

        ece = 0.0

        n = len(confidence)

        for i in range(bins):
            mask = (
                (confidence >= edges[i])
                &
                (confidence < edges[i + 1])
            )

            if mask.sum() == 0:
                continue
            acc = correctness[mask].mean()
            conf = confidence[mask].mean()

            ece += (
                abs(acc - conf)
                * mask.sum()
                / n
            )

        return float(ece)
    