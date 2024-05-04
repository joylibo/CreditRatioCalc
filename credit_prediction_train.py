from sqlmodel import SQLModel, Session, Field, select
from datetime import date
from typing import List, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from app.database import engine
from app.models.resident_credit_score import ResidentCreditScore, KeyResidentsActiveModel, PartyMemberPaymentsModel, CommunityServiceRecordModel, ElderlyServiceRecordModel, PartyMemberActivitiesModel, PartyMemberRewardsPunishmentsModel



"""
用于训练信用分预测模型
1. 从表`resident_credit_score_t`中读取用户信用分
2. 训练模型
3. 将模型保存到`credit_prediction_model.pkl`中
"""

# 根据日期范围获取用户信用分
def get_credit_scores(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session: Optional[Session] = None,
) -> List[ResidentCreditScore]:
    session = session or Session(engine)
    query = select(ResidentCreditScore)
    if start_date:
        query = query.where(ResidentCreditScore.day >= start_date)
    if end_date:
        query = query.where(ResidentCreditScore.day <= end_date)
    credit_scores = session.exec(query).all()
    session.close()
    return credit_scores

# 根据日期范围获取用户关键活动记录
def get_key_residents_active(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session: Optional[Session] = None,
) -> List[KeyResidentsActiveModel]:
    session = session or Session(engine)
    query = select(KeyResidentsActiveModel)
    if start_date:
        query = query.where(KeyResidentsActiveModel.last_active_date >= start_date)
    if end_date:
        query = query.where(KeyResidentsActiveModel.last_active_date <= end_date)
    key_residents_activates = session.exec(query).all()
    session.close()
    return key_residents_activates

# 根据日期范围获取用户党费缴纳记录
def get_party_member_payments(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session: Optional[Session] = None,
) -> List[PartyMemberPaymentsModel]:
    session = session or Session(engine)
    query = select(PartyMemberPaymentsModel)
    if start_date:
        query =query.where(PartyMemberPaymentsModel.payment_date >= start_date)
    if end_date:
        query = query.where(PartyMemberPaymentsModel.payment_date <= end_date)
    party_member_payments = session.exec(query).all()
    session.close()
    return party_member_payments

# 根据日期范围获取用户社区服务记录
def get_community_service_record(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session: Optional[Session] = None,
) -> List[CommunityServiceRecordModel]:
    session = session or Session(engine)
    query = select(CommunityServiceRecordModel)
    if start_date:
        query = query.where(CommunityServiceRecordModel.service_date >=start_date)
    if end_date:
        query = query.where(CommunityServiceRecordModel.service_date <= end_date)
    community_service_recaord = session.exec(query).all()
    session.close()
    return community_service_recaord

# 根据日期范围获取用户老人服务记录
def get_elderly_service_record(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session: Optional[Session] = None,
) -> List[ElderlyServiceRecordModel]:
    session = session or Session(engine)
    query = select(ElderlyServiceRecordModel)
    if start_date:
        query = query.where(ElderlyServiceRecordModel.evaluation_date >= start_date)
    if end_date:
        query = query.where(ElderlyServiceRecordModel.evaluation_date <= end_date)
    elderly_service_record = session.exec(query).all()
    session.close()
    return elderly_service_record

# 根据日期范围获取用户党活动记录
def get_party_member_activities(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session: Optional[Session] = None,
) -> List[PartyMemberActivitiesModel]:
    session = session or Session(engine)
    query = select(PartyMemberActivitiesModel)
    if start_date:
        query = query.where(PartyMemberActivitiesModel.add_day >= start_date)
    if end_date:
        query = query.where(PartyMemberActivitiesModel.add_day <= end_date)
    party_member_activities = session.exec(query).all()
    session.close()
    return party_member_activities

# 根据日期范围获取用户奖惩记录
def get_party_member_rewards_punishments(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session: Optional[Session] = None,
) -> List[PartyMemberRewardsPunishmentsModel]:
    session = session or Session(engine)
    query = select(PartyMemberRewardsPunishmentsModel)
    if start_date:
       query = query.where(PartyMemberRewardsPunishmentsModel.date_issued >= start_date)
    if end_date:
        query = query.where(PartyMemberRewardsPunishmentsModel.date_issued <= end_date)
    party_member_rewards_punishments = session.exec(query).all()
    session.close()
    return party_member_rewards_punishments


# 定义数据处理函数
def process_data(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    session: Optional[Session] = None,
) -> pd.DataFrame:
    # 获取各种数据
    credit_scores = get_credit_scores(start_date, end_date, session)
    key_residents_active = get_key_residents_active(start_date, end_date, session)
    party_member_payments = get_party_member_payments(start_date, end_date, session)
    community_service_record = get_community_service_record(start_date, end_date, session)
    elderly_service_record = get_elderly_service_record(start_date, end_date, session)
    party_member_activities = get_party_member_activities(start_date, end_date, session)
    party_member_rewards_punishments = get_party_member_rewards_punishments(start_date, end_date, session)

    # 将数据转换为 DataFrame
    data = pd.DataFrame({
        'credit_score': [score.value for score in credit_scores],
        'is_key_resident_active': [int(active.is_active) for active in key_residents_active],
        'party_fee_paid': [payment.amount for payment in party_member_payments],
        'community_service_hours': [record.hours for record in community_service_record],
        'elderly_service_rating': [record.rating for record in elderly_service_record],
        'party_activities_count': [len(activities) for activities in party_member_activities],
        'rewards_count': [len(rewards) for rewards in party_member_rewards_punishments],
        'punishments_count': [len(punishments) for punishments in party_member_rewards_punishments]
    })

    return data

# 定义模型训练函数
def train_model(data: pd.DataFrame, target_column: str, model_path: str):
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier())
    ])

    pipeline.fit(X, y)

    dump(pipeline, model_path)

# 示例用法
if __name__ == '__main__':
    start_date = date(2022, 1, 1)
    end_date = date(2022, 12, 31)
    data = process_data(start_date, end_date)
    train_model(data, 'credit_score', 'credit_score_model.pkl')