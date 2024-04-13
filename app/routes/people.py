from fastapi import APIRouter
from sqlmodel import Session, select
from app.database.database import engine
from app.models.resident_credit_score import ResidentCreditScore

router = APIRouter()

@router.get("/get_score")
def get_score():
    """返回若干条人群信用分测试数据库的连接
    """
    with Session(engine) as session:
        # 查询前10行数据
        query = select(ResidentCreditScore).limit(5)
        results = session.exec(query)
        return results.all()

@router.get("/get_score_by_id/{resident_id}")
def get_score_by_id(resident_id: int):
    return 'TBD'