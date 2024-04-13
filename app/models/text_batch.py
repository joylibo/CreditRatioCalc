from pydantic import BaseModel, Field
from typing import List

class Weights(BaseModel):
    cosine: float = 1.0
    euclidean: float = 0.0
    manhattan: float = 0.0

class TextBatch(BaseModel):
    reference_text: str = Field(
        default=None, title="计算相似度的目标文本", max_length=200
    )
    texts_to_compare: List[str] = Field(
        examples=[["str1", "str2"]],  
        title="待检测文本列表",
        description="当只有一个参数的时候，无论相似度是多少，占比都会是100%",
    )
    weights: Weights

class SimilarityScore(BaseModel):
    similarity: float
    rates: float