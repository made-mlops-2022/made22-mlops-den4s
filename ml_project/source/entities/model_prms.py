from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    model_type: str = field(default="RandomForestClassifier")
    n_estimators: int = field(default=50)  # for RandomForestClassifier
    max_depth: int = field(default=10)
    random_state: int = field(default=32)
    inv_reg_strength: float = field(default=1.0)  # for LogisticRegression
    intercept_scaling: float = field(default=1.0)
