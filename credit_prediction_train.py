from sqlmodel import SQLModel, Session, Field, select
from datetime import datetime, date
from app.database import engine

"""
用于训练信用分预测模型
1. 从表`resident_credit_score_t`中读取用户信用分
2. 训练模型
3. 将模型保存到`credit_prediction_model.pkl`中
"""

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

# 创建会话
with Session(engine) as session:
    # 查询前10行数据
    query = select(ResidentCreditScore).limit(10)
    results = session.exec(query)

    # 处理查询结果
    for result in results:
        print(result)