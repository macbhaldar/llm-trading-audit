from sklearn.metrics import (

    mean_absolute_error,

    mean_squared_error,

    r2_score,
)

import numpy as np


class XGBoostEvaluator:

    @staticmethod
    def evaluate(y_true, predictions):

        mse = mean_squared_error(

            y_true,

            predictions,
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(

            y_true,

            predictions,
        )

        r2 = r2_score(

            y_true,

            predictions,
        )

        return {

            "RMSE": rmse,

            "MAE": mae,

            "R2": r2,
        }