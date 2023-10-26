from pydantic import BaseModel


class EvalResults(BaseModel):
    total_examples: int
    recall: float
    precision: float
    hit_rate: float
