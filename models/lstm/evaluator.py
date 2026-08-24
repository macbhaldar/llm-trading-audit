import numpy as np

from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score
)


class LSTMEvaluator:

    @staticmethod
    def evaluate(

        y_true,

        prediction

    ):

        mse = mean_squared_error(

            y_true,

            prediction
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(

            y_true,

            prediction
        )

        r2 = r2_score(

            y_true,

            prediction
        )

        return {

            "RMSE": rmse,

            "MAE": mae,

            "R2": r2
        }
    