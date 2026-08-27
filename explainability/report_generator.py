import json
from pathlib import Path


class ExplainabilityReport:

    @staticmethod
    def save(
        report,
        path,
    ):

        Path(path).parent.mkdir(

            parents=True,

            exist_ok=True,
        )

        with open(
            path,
            "w",
        ) as f:

            json.dump(
                report,
                f,
                indent=4,
            )