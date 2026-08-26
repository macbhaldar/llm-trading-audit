import numpy as np


class ConsistencyChecker:

    @staticmethod
    def agreement(predictions):

        if len(predictions) == 0:

            return 0

        values, counts = np.unique(
            predictions,
            return_counts=True
        )

        return counts.max() / len(predictions)
    