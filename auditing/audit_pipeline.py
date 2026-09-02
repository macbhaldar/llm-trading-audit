from .calibration import Calibration
from .hallucination import HallucinationDetector
from .theory_validator import TheoryValidator
from .trust_score import TrustScore


class AuditPipeline:
    """
    Runs all audit modules for one prediction.
    """
    def audit(
        self,
        prediction,
        actual_correct,
        rsi,
        sentiment,
        explainability=1.0,
    ):

        calibration = abs(
            prediction["Confidence"]
            -
            actual_correct
        )
        hallucination = HallucinationDetector.score(
            prediction["Reasoning"]
        )

        theory = TheoryValidator.validate(
            prediction["Recommendation"],
            rsi,
            sentiment
        )

        trust = TrustScore.compute(
            accuracy=float(actual_correct),
            calibration=calibration,
            hallucination=hallucination,
            risk=0.10,
            theory=theory,
            explainability=explainability
        )

        return {
            "CalibrationError": calibration,
            "HallucinationScore": hallucination,
            "TheoryScore": theory,
            "TrustScore": trust
        }
    