import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

        # 确保序列长度一致
        if len(seq_data) < self.seq_length:
            pad_length = self.seq_length - len(seq_data)
            seq_data = np.pad(seq_data, ((pad_length, 0), (0, 0)), 'constant', constant_values=0)
        
        # 确保目标长度一致
        if len(target_data) < self.pred_length:
            pad_length = self.pred_length - len(target_data)
            target_data = np.pad(target_data, (0, pad_length), 'constant', constant_values=0)
        
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
num_epochs = 50  # 增加训练轮数
patience = 5  # 早停的耐心参数

# 数据分割
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# 创建训练集和验证集的数据加载器
train_dataset = CreditScoreDataset(train_data, seq_length, pred_length)
val_dataset = CreditScoreDataset(val_data, seq_length, pred_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CreditScoreRNN(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
print("Training model...")
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")
    
    # 验证模型
    model.eval()
    val_loss = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    val_loss /= len(val_loader)
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    mse = mean_squared_error(all_targets, all_outputs)
    mae = mean_absolute_error(all_targets, all_outputs)
    print(f"Validation Loss: {val_loss}, MSE: {mse}, MAE: {mae}")

    # 早停策略
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最优模型
        torch.save(model.state_dict(), './best_credit_score_rnn_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# 保存最终模型
model_path = './final_credit_score_rnn_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
