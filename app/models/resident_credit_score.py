from sqlmodel import Field as Field, SQLModel
from datetime import datetime, date

# 定义模型类
class ResidentCreditScore(SQLModel, table=True):
    __tablename__ = "resident_credit_score_t"

    id: int = Field(default=None, primary_key=True)
    account_id: int
    primary_id: int
    create_time: datetime
    update_time: datetime
    day: date
    score: float
    resident_id: int
    reason: str
    credit_assessment: str