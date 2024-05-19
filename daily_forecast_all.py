import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sqlmodel import Session, select
from app.database.database import engine
from datetime import datetime, timedelta
from app.models.resident_credit_score import ResidentCreditScore, ResidentCreditTrendModel
from tqdm import tqdm

# 定义SQLModel模型
from sqlmodel import SQLModel

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

def predict_future_scores(model, input_scores):
    input_tensor = torch.tensor(input_scores, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output.cpu().numpy().flatten()

# 配置
# resident_id = 25313  # 替换为实际的resident_id
# primary_id = 1     # 替换为实际的primary_id
model_path = 'best_credit_score_lstm_model.pth'

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device)
model.load_state_dict(torch.load(model_path))


# 清空数据库中的ResidentCreditTrendModel
def clear_database():
    with Session(engine) as session:
        statement = select(ResidentCreditTrendModel)
        results = session.exec(statement).all()
        for result in results:
            session.delete(result)
        session.commit()

# 通过引入的ResidentCreditTrendModel类，把future_scores数据写入数据库

def post_future_scores(resident_id, primary_id, account_id, future_scores):
    with Session(engine) as session:
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
            session.add(trend)
            session.commit()
            # print(f"Posted future score {score} for day {datetime.now() + timedelta(days=i+1)}")

# 获取全部的resident_id, 对于这些resident_id分别调用get_recent_scores函数获取100天的得分
def get_all_resident_ids():
    with Session(engine) as session:
        statement = select(ResidentCreditScore.resident_id).distinct()
        results = session.exec(statement).all()
    return [result for result in results]

def get_all_primary_ids(resident_id):
    with Session(engine) as session:
        statement = select(ResidentCreditScore.primary_id).where(ResidentCreditScore.resident_id == resident_id).distinct()
        results = session.exec(statement).all()
    return [result for result in results]


if __name__ == '__main__':
    clear_database() # 每一次执行都先清空预测表
    account_id = 2
    all_resident_ids = get_all_resident_ids()
    for resident_id in tqdm(all_resident_ids):
        all_primary_ids = get_all_primary_ids(resident_id)
        for primary_id in all_primary_ids:
            recent_scores = get_recent_scores(resident_id, primary_id)
            if len(recent_scores) < 100:
                print(f"Not enough data for prediction for resident_id={resident_id}, primary_id={primary_id}, account_id={account_id}")
            else:
                future_scores = predict_future_scores(model, recent_scores)
                post_future_scores(resident_id, primary_id, account_id, future_scores)
                

