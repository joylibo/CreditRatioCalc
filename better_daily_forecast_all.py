import pandas as pd
import torch
import torch.nn as nn
from sqlmodel import Session, select
from app.database.database import engine
from datetime import datetime, timedelta
from app.models.resident_credit_score import ResidentCreditScore, ResidentCreditTrendModel
from tqdm import tqdm
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# 配置模型路径
model_path = 'best_credit_score_lstm_model.pth'

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device)
model.load_state_dict(torch.load(model_path))

# 清空指定resident_id和primary_id的预测数据
def clear_resident_data(resident_id, primary_id):
    with Session(engine) as session:
        delete_statement = (
            ResidentCreditTrendModel.__table__.delete()
            .where(ResidentCreditTrendModel.resident_id == resident_id)
            .where(ResidentCreditTrendModel.primary_id == primary_id)
        )
        session.execute(delete_statement)
        session.commit()

# 批量写入未来预测分数
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

def use_tqdm():
    # 检查是否在终端运行
    return os.isatty(sys.stdout.fileno())

# 处理单个resident_id的函数
def process_resident(resident_id, account_id):
    all_primary_ids = get_all_primary_ids(resident_id)
    for primary_id in all_primary_ids:
        recent_scores = get_recent_scores(resident_id, primary_id)
        if len(recent_scores) < 100:
            print(f"Not enough data for prediction for resident_id={resident_id}, primary_id={primary_id}, account_id={account_id}")
        else:
            clear_resident_data(resident_id, primary_id)  # 清除旧数据
            future_scores = predict_future_scores(model, recent_scores)
            post_future_scores(resident_id, primary_id, account_id, future_scores)

if __name__ == '__main__':
    account_id = 2
    all_resident_ids = get_all_resident_ids()
    tqdm_func = tqdm if use_tqdm() else lambda x: x  # 如果在终端运行则使用tqdm，否则使用原始迭代器

    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_resident, resident_id, account_id): resident_id for resident_id in tqdm_func(all_resident_ids)}
        for future in as_completed(futures):
            resident_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f'Resident ID {resident_id} generated an exception: {exc}')
