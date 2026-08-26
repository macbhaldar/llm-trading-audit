import pandas as pd


class ModelComparison:

    @staticmethod
    def compare(results):

        return (

            pd.DataFrame(results)

            .T

            .sort_values(

                "RMSE",

                ascending=True,

            )

        )
