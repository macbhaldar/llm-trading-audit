import pandas as pd
import numpy as np

class FeatureImportance:

    @staticmethod
    def from_tree_model(model):
        return (
            pd.DataFrame(
                {
                    "Feature":
                    model.feature_names_in_,

                    "Importance":
                    model.feature_importances_,
                }
            )
            .sort_values(
                "Importance",
                ascending=False,
            )
        )

    @staticmethod
    def permutation(
        feature_names,
        scores,
    ):

        return pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": scores,
            }

        ).sort_values(
            "Importance",
            ascending=False,
        )
    