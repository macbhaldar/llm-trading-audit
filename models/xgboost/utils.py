import pandas as pd


def feature_importance(model):

    importance = model.model.feature_importances_

    names = model.model.feature_names_in_

    return pd.DataFrame(

        {

            "Feature": names,

            "Importance": importance,

        }

    ).sort_values(

        "Importance",

        ascending=False,
    )