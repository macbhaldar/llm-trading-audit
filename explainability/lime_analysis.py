from lime.lime_tabular import LimeTabularExplainer


class LIMEExplainer:
    def __init__(
        self,
        train_data,
        feature_names,
        mode="regression",
    ):

        self.explainer = LimeTabularExplainer(
            training_data=train_data,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True,
        )

    def explain(
        self,
        instance,
        predict_fn,
        num_features=10,

    ):

        return self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
        )
