from typing import List, Union

from pydantic import BaseModel, conlist, validator

from .feature_prms import ALL_FEATURES


class AppRequest(BaseModel):
    data: List[conlist(Union[float, str], min_items=1)]  # max_items=50
    features: List[str]

    @validator("features")
    def validate_model_features(cls, features):
        if features != ALL_FEATURES:
            raise ValueError(f"incorrect features number or (and) order:\n" +
                             f"expected: {ALL_FEATURES}")
        return features


class AppResponse(BaseModel):
    id: str
    disease: int
