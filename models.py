from pydantic import BaseModel, Field, constr, validator
from sqlmodel import Field as modelField, SQLModel
from typing import List, Dict
from enum import Enum
from datetime import datetime, date

# 定义模型类
class ResidentCreditScore(SQLModel, table=True):
    __tablename__ = "resident_credit_score_t"

    id: int = modelField(default=None, primary_key=True)
    account_id: int
    primary_id: int
    create_time: datetime
    update_time: datetime
    day: date
    score: float
    resident_id: int
    reason: str
    credit_assessment: str


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

# 定义势态感知的request结构
class SceneName(str, Enum):
    SHE_QU = "平安社区"
    DANG_JIAN = "信用党建"
    YANG_LAO = "信用养老"

class CurrentSceneRequest(BaseModel):
    scene_name: SceneName
    perception_deadline: datetime

# 定义势态感知的返回结构
class ScenePerceptionResponse(BaseModel):
    current_scene: SceneName
    scene_perception: Dict[str, float]

# 定义个人信用值预测的request结构
class CreditPredictionRequest(BaseModel):
    scene_name: SceneName
    resident_id: int
    prediction_time: datetime
# 定义个人信用值预测的response结构
class CreditPredictionResponse(BaseModel):
    credit_score: float

# 定义分组信用值预测的request结构
class CreditPredictionByGroupRequest(BaseModel):
    scene_name: SceneName
    group_name: str
    prediction_time: datetime

# 定义分组信用值预测的response结构
class CreditPredictionByGroupResponse(BaseModel):
    credit_score: float

# 定义个人信用值预警的request结构
class CreditWarningRequest(BaseModel):
    scene_name: SceneName # 场景名称
    warning_time: datetime # 预警时间
    warning_threshold: int = 70  # 预警阈值

# 定义个人信用值预警的response结构
class CreditWarningResponse(BaseModel):
    resident_id: int  # 预警人员ID
    resident_name: str  # 预警人员姓名
    credit_score: float  # 预警人员信用分值
