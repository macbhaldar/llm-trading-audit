import json
from pathlib import Path


class EvaluationReport:

    @staticmethod
    def save(
        metrics,
        path,
    ):

        Path(path).parent.mkdir(
            exist_ok=True,
            parents=True,
        )

        with open(
            path,
            "w",
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4,
            )
