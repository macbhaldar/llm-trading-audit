from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class ClassificationMetrics:

    @staticmethod
    def evaluate(y_true, y_pred):

        return {

            "Accuracy":
            accuracy_score(y_true, y_pred),

            "Precision":
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "Recall":
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "F1":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        }
