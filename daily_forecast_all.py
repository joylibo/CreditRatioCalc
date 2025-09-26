# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
from sqlmodel import Session, select
from app.database.database import engine
from datetime import datetime, timedelta
from app.models.resident_credit_score import ResidentCreditScore, ResidentCreditScorePlus, ResidentCreditTrendModel
from tqdm import tqdm
import sys
import os

# 设置工作目录为脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=30, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 100, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def get_recent_scores(resident_id, primary_id, days=100):
    with Session(engine) as session:
        statement = (
            select(ResidentCreditScore.score, ResidentCreditScore.day)
            .where(ResidentCreditScore.resident_id == resident_id)
            .where(ResidentCreditScore.primary_id == primary_id)
            .order_by(ResidentCreditScore.day.desc())
            .limit(days)
        )
        results = session.exec(statement).all()

    # 将数据转为DataFrame并按照日期排序
    df = pd.DataFrame(results)
    df['day'] = pd.to_datetime(df['day'])
    df.sort_values(by='day', inplace=True)
    return df['score'].values



def get_recent_score_v2(resident_id, table_model, primary_id, days=100):
    if table_model is None:
        return [], None

    with Session(engine) as session:
        statement = (
            select(table_model.score, table_model.day, table_model.account_id)
            .where(table_model.resident_id == resident_id)
            .where(table_model.primary_id == primary_id)
            .order_by(table_model.day.desc())
            .limit(days)
        )
        results = session.exec(statement).all()

    # 将数据转为DataFrame并按照日期排序
    df = pd.DataFrame(results)
    df['day'] = pd.to_datetime(df['day'])
    df.sort_values(by='day', inplace=True)

    # 提取account_id（取第一个，假设所有记录的account_id相同）
    account_id = df['account_id'].iloc[0] if not df.empty else None

    return df['score'].values, account_id

def predict_future_scores(model, input_scores):
    input_tensor = torch.tensor(input_scores, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output.cpu().numpy().flatten()

# 配置模型路径
model_path = './best_credit_score_lstm_model.pth'

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device)
model.load_state_dict(torch.load(model_path))


# Deprecation Warning: 清空数据库中的ResidentCreditTrendModel
def clear_database():
    with Session(engine) as session:
        statement = select(ResidentCreditTrendModel)
        results = session.exec(statement).all()
        for result in results:
            session.delete(result)
        session.commit()

# 清空指定resident_id和primary_id的预测数据
def clear_resident_data(resident_id, primary_id):
    with Session(engine) as session:
        delete_statement = (
            ResidentCreditTrendModel.__table__.delete()
            .where(ResidentCreditTrendModel.resident_id == resident_id)
            .where(ResidentCreditTrendModel.primary_id == primary_id)
        )
        session.exec(delete_statement)
        session.commit()

# 通过引入的ResidentCreditTrendModel类，把future_scores数据写入数据库，批量写入
def post_future_scores(resident_id, primary_id, account_id, future_scores):
    trends = []
    for i, score in enumerate(future_scores):
        trend = ResidentCreditTrendModel(
            resident_id=resident_id,
            primary_id=primary_id,
            score=score,
            day=datetime.now() + timedelta(days=i+1),
            account_id=account_id,
            create_time=datetime.now(),
            update_time=datetime.now(),
            current_score=score,
            reason='Predicted'
        )
        trends.append(trend)

    with Session(engine) as session:
        session.bulk_save_objects(trends)
        session.commit()

# 获取全部的resident_id，从两个表中获取所有唯一的居民ID及其对应的表
def get_all_resident_ids():
    with Session(engine) as session:
        # 从原表获取
        statement1 = select(ResidentCreditScore.resident_id).distinct()
        results1 = session.exec(statement1).all()

        # 从新表获取
        statement2 = select(ResidentCreditScorePlus.resident_id).distinct()
        results2 = session.exec(statement2).all()

        # 创建居民到表的映射字典
        resident_tables = {}
        for rid in results1:
            resident_tables[rid] = ResidentCreditScore
        for rid in results2:
            resident_tables[rid] = ResidentCreditScorePlus

        return resident_tables

def get_all_primary_ids(resident_id, table_model):
    if table_model is None:
        return []

    with Session(engine) as session:
        statement = select(table_model.primary_id).where(table_model.resident_id == resident_id).distinct()
        results = session.exec(statement).all()
    return [result for result in results]

def use_tqdm():
    # 检查是否在终端运行
    return os.isatty(sys.stdout.fileno())

if __name__ == '__main__':
    print("开始批量预测信用分数...")
    print("从两个表中获取所有居民ID...")
    resident_tables = get_all_resident_ids()
    print(f"共找到 {len(resident_tables)} 个居民")

    tqdm_func = tqdm if use_tqdm() else lambda x: x  # 如果在终端运行则使用tqdm，否则使用原始迭代器
    processed_count = 0
    skipped_count = 0

    for resident_id, table_model in tqdm_func(resident_tables.items()):
        all_primary_ids = get_all_primary_ids(resident_id, table_model)
        for primary_id in all_primary_ids:
            recent_scores, account_id = get_recent_score_v2(resident_id, table_model, primary_id)
            if len(recent_scores) < 100:
                print(f"数据不足: resident_id={resident_id}, primary_id={primary_id}, account_id={account_id}, 数据点数={len(recent_scores)}")
                skipped_count += 1
            else:
                clear_resident_data(resident_id, primary_id)  # 清除旧数据
                future_scores = predict_future_scores(model, recent_scores)
                post_future_scores(resident_id, primary_id, account_id, future_scores)
                processed_count += 1

    print(f"预测完成! 成功处理: {processed_count} 组数据, 跳过: {skipped_count} 组数据")
