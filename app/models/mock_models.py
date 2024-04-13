from fastapi import BaseModel, Enum
from typing import Dict
from datetime import datetime


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