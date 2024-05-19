import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# 1. 加载并预处理数据
print("Loading and preprocessing data...")

file_path = './历史行为与信用数据.xlsx'

data = pd.read_excel(file_path)
data2 = pd.read_excel(file_path, sheet_name='Sheet2')
data3 = pd.read_excel(file_path, sheet_name='Sheet3')
data = pd.concat([data, data2, data3], ignore_index=True)

print(f"数据集总共有 {len(data)} 行")

# 确保日期字段是日期类型
data['record_date'] = pd.to_datetime(data['record_date'])

# 按照 resident_id, primary_id, record_date 排序
data.sort_values(by=['resident_id', 'primary_id', 'record_date'], inplace=True)

# 2. 构建数据集
class CreditScoreDataset(Dataset):
    def __init__(self, data, input_days=100, output_days=30):
        self.data = data
        self.input_days = input_days
        self.output_days = output_days
        self.residents = data['resident_id'].unique()
        self.primary_ids = data['primary_id'].unique()
        self.sequence_data = self.create_sequences()

    def create_sequences(self):
        sequences = []
        for resident in self.residents:
            for primary_id in self.primary_ids:
                resident_data = self.data[(self.data['resident_id'] == resident) & (self.data['primary_id'] == primary_id)]
                if len(resident_data) >= self.input_days + self.output_days:
                    for start in range(len(resident_data) - self.input_days - self.output_days + 1):
                        input_seq = resident_data.iloc[start:start + self.input_days]['credit_score'].values
                        output_seq = resident_data.iloc[start + self.input_days:start + self.input_days + self.output_days]['credit_score'].values
                        sequences.append((input_seq, output_seq))
        return sequences

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        input_seq, output_seq = self.sequence_data[idx]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

input_days = 100
output_days = 30
dataset = CreditScoreDataset(data, input_days=input_days, output_days=output_days)

# 数据分割：80%训练集，20%验证集
train_size = int(0.8 * len(dataset))
train_dataset = Subset(dataset, range(train_size))
val_dataset = Subset(dataset, range(train_size, len(dataset)))

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 3. 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=output_days, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * input_days, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 100
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        inputs, targets = inputs.unsqueeze(-1).to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_dataloader)

    # 验证模型
    model.eval()
    val_loss = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.unsqueeze(-1).to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    avg_val_loss = val_loss / len(val_dataloader)
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    mape = np.mean(np.abs((all_targets - all_predictions) / all_targets)) * 100

    print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}%')

    # 早停机制
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_credit_score_lstm_model.pth')
        print('Best model saved.')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping triggered.')
            break

print('Training complete.')
