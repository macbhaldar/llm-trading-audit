from sklearn.model_selection import train_test_split

from .model import XGBoostModel


class XGBoostTrainer:

    def __init__(self, config):

        self.model = XGBoostModel(config)

    def train(

        self,

        X,

        y,

        test_size=0.20,

        random_state=42,

    ):

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=test_size,

            random_state=random_state,

        )

        self.model.fit(X_train, y_train)

        return (

            self.model,

            X_train,

            X_test,

            y_train,

            y_test,
        )