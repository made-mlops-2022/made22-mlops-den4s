from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    continuous_features: List[str]
    discrete_features: List[str]
    target_column: Optional[str]
