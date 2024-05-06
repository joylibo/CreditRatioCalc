from sqlmodel import SQLModel, Session, Field, select
from datetime import date
from typing import List, Optional, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from tqdm import tqdm
from app.database.database import engine
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

def get_all_data(start_date: date, end_date: date, session: Optional[Session] = None) -> pd.DataFrame:
    credit_scores = get_credit_scores(start_date, end_date, session)
    key_residents_active = get_key_residents_active(start_date, end_date, session)
    party_member_payments = get_party_member_payments(start_date, end_date, session)
    community_service_record = get_community_service_record(start_date, end_date, session)
    elderly_service_record = get_elderly_service_record(start_date, end_date, session)
    party_member_activities = get_party_member_activities(start_date, end_date, session)
    party_member_rewards_punishments = get_party_member_rewards_punishments(start_date, end_date, session)

    data = []
    for score in tqdm(credit_scores, desc='Processing data'):
        resident_id = score.resident_id
        score_date = score.day
        is_key_resident_active = any(active.last_active_date >= start_date and active.last_active_date <= end_date for active in key_residents_active if active.resident_id == resident_id)
        party_fee_paid = sum(payment.payment_amount for payment in party_member_payments if payment.resident_id == resident_id and payment.payment_date >= start_date and payment.payment_date <= end_date)
        community_service_hours = sum(record.service_score for record in community_service_record if record.resident_id == resident_id and record.service_date >= start_date and record.service_date <= end_date)
        relevant_elderly_records = [record for record in elderly_service_record if record.resident_id == resident_id and record.evaluation_date >= start_date and record.evaluation_date <= end_date]
        elderly_service_rating = sum([record.caregiving_knowledge + record.nursing_knowledge + record.life_care_ability + record.basic_nursing_ability + record.specialized_nursing_ability + record.cultural_nursing_ability + record.professional_skills + record.professional_attitude + record.personal_characteristics for record in relevant_elderly_records]) / (9.0 * len(relevant_elderly_records)) if relevant_elderly_records else 0
        party_activities_count = len([activity for activity in party_member_activities if activity.resident_id == resident_id and activity.add_day >= start_date and activity.add_day <= end_date])
        rewards_count = len([reward for reward in party_member_rewards_punishments if reward.resident_id == resident_id and reward.type == '奖励' and reward.date_issued >= start_date and reward.date_issued <= end_date])
        punishments_count = len([punishment for punishment in party_member_rewards_punishments if punishment.resident_id == resident_id and punishment.type == '处罚' and punishment.date_issued >= start_date and punishment.date_issued <= end_date])

        data.append({
            'resident_id': resident_id,
            'score_date': score_date,
            'credit_score': score.score,
            'is_key_resident_active': int(is_key_resident_active),
            'party_fee_paid': party_fee_paid,
            'community_service_hours': community_service_hours,
            'elderly_service_rating': elderly_service_rating,
            'party_activities_count': party_activities_count,
            'rewards_count': rewards_count,
            'punishments_count': punishments_count
        })

    data = pd.DataFrame(data)
    return data

# 定义数据处理函数
def process_data(
    train_start_date: date, 
    train_end_date: date,
    val_start_date: date,
    val_end_date: date,
    session: Optional[Session] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # 获取训练集数据
    train_data = get_all_data(train_start_date, train_end_date, session)

    # 获取验证集数据
    val_data = get_all_data(val_start_date, val_end_date, session)

    return train_data, val_data


# 定义模型训练和评估函数
def train_and_evaluate_model(
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    target_column: str, 
    model_path: str
):
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    X_val = val_data.drop(target_column, axis=1)
    y_val = val_data[target_column]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier())
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation accuracy: {val_accuracy:.4f}')

    dump(pipeline, model_path)

# 示例用法
if __name__ == '__main__':
    train_start_date = date(2023, 4, 17)
    train_end_date = date(2023, 12, 31)
    val_start_date = date(2024, 1, 1)
    val_end_date = date(2024, 5, 4)

    train_data, val_data = process_data(train_start_date, train_end_date, val_start_date, val_end_date)
    train_and_evaluate_model(train_data, val_data, 'credit_score', 'credit_score_model.pkl')