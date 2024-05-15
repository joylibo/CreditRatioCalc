import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

# 数据集类
class CreditScoreDataset(Dataset):
    def __init__(self, data, seq_length=100, pred_length=30):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.resident_ids = data['resident_id'].unique()

    def __len__(self):
        return len(self.resident_ids)

    def __getitem__(self, idx):
        resident_id = self.resident_ids[idx]
        resident_data = self.data[self.data['resident_id'] == resident_id]
        resident_data = resident_data.sort_values(by='record_date')

        seq_data = resident_data.iloc[-self.seq_length-self.pred_length:-self.pred_length][['credit_score', 'service_score', 'ability_score', 'evaluation_duration', 'party_act_duration', 'pay_amount']].values
        target_data = resident_data.iloc[-self.pred_length:]['credit_score'].values

        return torch.tensor(seq_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32)

# RNN模型
class CreditScoreRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CreditScoreRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h_0)
        out = self.fc(out[:, -1, :])
        return out

# 读取并预处理数据
print("Loading and preprocessing data...")
data = pd.read_excel('./历史信用及行为数据.xlsx')
data.fillna(0, inplace=True)  # 简单处理缺失值

# 设置模型参数
seq_length = 100
pred_length = 30
input_size = 6  # 包括 'credit_score', 'service_score', 'ability_score', 'evaluation_duration', 'party_act_duration', 'pay_amount'
hidden_size = 64
output_size = pred_length
num_layers = 1
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# 创建数据集和数据加载器
dataset = CreditScoreDataset(data, seq_length, pred_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CreditScoreRNN(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
print("Training model...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")

# 保存模型
model_path = './credit_score_rnn_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
