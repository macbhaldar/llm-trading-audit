class TrustScore:

    WEIGHTS = {

        "accuracy": 0.30,

        "calibration": 0.20,

        "hallucination": 0.15,

        "risk": 0.15,

        "theory": 0.10,

        "explainability": 0.10,
    }

    @classmethod
    def compute(

        cls,

        accuracy,

        calibration,

        hallucination,

        risk,

        theory,

        explainability,
    ):

        return round(

            100
            * (

                accuracy
                * cls.WEIGHTS["accuracy"]

                +

                (1 - calibration)
                * cls.WEIGHTS["calibration"]

                +

                (1 - hallucination)
                * cls.WEIGHTS["hallucination"]

                +

                (1 - risk)
                * cls.WEIGHTS["risk"]

                +

                theory
                * cls.WEIGHTS["theory"]

                +

                explainability
                * cls.WEIGHTS["explainability"]

            ),

            2
        )