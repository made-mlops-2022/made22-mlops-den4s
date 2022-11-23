from dataclasses import dataclass
from typing import List, Optional


ALL_FEATURES = ["age", "sex", "cp", "trestbps", "chol",
                "fbs",   "restecg",  "thalach",  "exang",    "oldpeak",
                "slope", "ca",       "thal",     "id"]


@dataclass()
class FeatureParams:
    continuous_features: List[str]
    discrete_features: List[str]
    target_col: Optional[str]
