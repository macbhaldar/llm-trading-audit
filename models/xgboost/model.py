import xgboost as xgb


class XGBoostModel:

    def __init__(self, config):

        self.model = xgb.XGBRegressor(

            objective=config.objective,

            n_estimators=config.n_estimators,

            learning_rate=config.learning_rate,

            max_depth=config.max_depth,

            min_child_weight=config.min_child_weight,

            subsample=config.subsample,

            colsample_bytree=config.colsample_bytree,

            gamma=config.gamma,

            reg_alpha=config.reg_alpha,

            reg_lambda=config.reg_lambda,

            random_state=config.random_state,

            n_jobs=-1,
        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)

    def save(self, path):

        self.model.save_model(path)

    def load(self, path):

        self.model.load_model(path)