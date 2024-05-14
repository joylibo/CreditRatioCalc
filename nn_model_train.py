import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import torch
import torch.nn as nn

# 读取数据
print('读取数据')
data = pd.read_excel('./历史信用及行为数据.xlsx')
data = data.dropna(subset=['credit_score'])  # 删除目标变量为空的行
print('done')

# 分离数值型和非数值型特征
print('分离数值型和非数值型特征')
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns
print('done')

# 填充数值型特征的空值
print('填充数值型特征的空值')
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
print('done')

# 填充非数值型特征的空值
print('填充非数值型特征的空值')
for col in tqdm(non_numeric_cols, desc='Processing data'):
    data[col] = data[col].fillna(data[col].mode()[0])
print('done')

# 对文本类型的特征进行编码
print('对文本类型的特征进行编码')
cat_cols = ['party_mark_type', 'key_desc', 'pay_status']
data = pd.get_dummies(data, columns=cat_cols)
print('done')

# 按resident_id和record_date排序
print('按resident_id和record_date排序')
data = data.sort_values(by=['resident_id', 'record_date'])
print('done')

# 更新数值型特征列（包括编码后的文本特征）
numeric_cols = data.select_dtypes(include=['float64', 'int64', 'uint8']).columns

# 构造滑动窗口数据生成器
def generate_windows(group, window_size, future_days):
    for i in range(window_size, len(group) - future_days + 1):
        X = group.iloc[i-window_size:i, 2:].values
        y = group.iloc[i:i+future_days, 2].values.flatten()
        yield X, y

def data_generator(data, window_size, future_days):
    for _, group in data.groupby('resident_id'):
        for X, y in generate_windows(group, window_size, future_days):
            yield X, y

# 定义模型
class CreditScoreModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(CreditScoreModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.conv1d(x)
        x = self.pool1d(x)
        x = x.transpose(1, 2)  # 转置为 (batch_size, seq_len, features)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc(hn[-1])
        return out

# 实例化模型
input_size = len(numeric_cols) - 2  # 减去 'resident_id' 和 'record_date'
hidden_size = 128
output_size = 30  # 预测未来30天

model = CreditScoreModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
print('训练模型')
num_epochs = 50
window_size = 100
future_days = 30

# 统计总窗口数量
total_windows = sum(len(group) - window_size - future_days + 1 for _, group in data.groupby('resident_id'))

for epoch in range(num_epochs):
    epoch_loss = 0
    with tqdm(total=total_windows, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
        for X_batch, y_batch in data_generator(data, window_size, future_days):
            X_batch = torch.from_numpy(np.array(X_batch, dtype=np.float32))
            y_batch = torch.from_numpy(np.array(y_batch, dtype=np.float32))
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.update(1)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
print('done')

# 评估模型
print('评估模型')
X_test, y_test = [], []
for X_sample, y_sample in data_generator(data, window_size, future_days):
    X_test.append(X_sample)
    y_test.append(y_sample)

X_test = torch.from_numpy(np.array(X_test, dtype=np.float32))
y_test = torch.from_numpy(np.array(y_test, dtype=np.float32))

with torch.no_grad():
    y_pred = model(X_test)
    rmse = mean_squared_error(y_test.numpy(), y_pred.numpy(), squared=False)
    print(f'RMSE: {rmse}')
print('done')

# 保存模型
print('保存模型')
torch.save(model.state_dict(), 'credit_score_model.pth')
print('done')
