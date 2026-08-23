from dataclasses import dataclass


@dataclass
class XGBoostConfig:

    objective: str = "reg:squarederror"

    n_estimators: int = 500

    learning_rate: float = 0.05

    max_depth: int = 8

    min_child_weight: int = 3

    subsample: float = 0.80

    colsample_bytree: float = 0.80

    gamma: float = 0.10

    reg_alpha: float = 0.10

    reg_lambda: float = 1.0

    random_state: int = 42