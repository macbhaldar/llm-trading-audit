import shap
import pandas as pd


class SHAPExplainer:

    def __init__(self, model):

        self.model = model

        self.explainer = shap.Explainer(model.predict)

    def explain(self, X):

        return self.explainer(X)

    def feature_importance(self, X):

        values = self.explainer(X)

        importance = abs(values.values).mean(axis=0)

        return (

            pd.DataFrame(
                {
                    "Feature": X.columns,
                    "Importance": importance,
                }
            )
            .sort_values(
                "Importance",
                ascending=False,
            )
        )