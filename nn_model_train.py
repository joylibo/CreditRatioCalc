import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 读取数据
print('读取数据')
data = pd.read_excel('./历史信用及行为数据.xlsx')
data = data.dropna(subset=['credit_score'])  # 删除目标变量为空的行
print('done')

# 定义数据预处理函数
def preprocess_data(data):
    # 分离数值型和非数值型特征
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns

    # 填充数值型特征的空值
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # 填充非数值型特征的空值
    for col in tqdm(non_numeric_cols, desc='Processing data'):
        data[col] = data[col].fillna(data[col].mode()[0])

    # 对文本类型的特征进行编码
    cat_cols = ['party_mark_type', 'key_desc', 'pay_status']
    data = pd.get_dummies(data, columns=cat_cols)

    # 按resident_id和record_date排序
    data = data.sort_values(by=['resident_id', 'record_date'])

    return data

data = preprocess_data(data)

# 构造滑动窗口数据生成器
class CreditDataset(Dataset):
    def __init__(self, data, window_size, future_days):
        self.data = data
        self.window_size = window_size
        self.future_days = future_days

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        group = self.data.iloc[idx]
        X, y = self.generate_windows(group, self.window_size, self.future_days)  # 传递 future_days 参数
        return X, y

    def generate_windows(self, group, window_size, future_days):
        X, y = [], []
        for i in range(len(group) - window_size - future_days + 1):
            X.append(group.iloc[i:i+window_size, 2:].values)
            y.append(group.iloc[i+window_size:i+window_size+future_days, 2].values.flatten())
        return np.array(X), np.array(y)



# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = CreditDataset(train_data, window_size=100, future_days=30)
test_dataset = CreditDataset(test_data, window_size=100, future_days=30)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型
class CreditScoreModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(CreditScoreModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.fc(hn[-1])
        return out


# 使用 GPU 进行加速计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 实例化模型
input_size = len(data.columns) - 2  # 减去 'resident_id' 和 'record_date'
hidden_size = 128
output_size = 30  # 预测未来30天的信用分数

model = CreditScoreModel(input_size, hidden_size, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
print('训练模型')
num_epochs = 50

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.float().to(device)
        y_batch = y_batch.float().to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 评估模型
print('评估模型')
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test = X_test.float().to(device)
        y_test = y_test.float().to(device)

        outputs = model(X_test)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(y_test.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
rmse = mean_squared_error(all_targets, all_preds, squared=False)
print(f'RMSE: {rmse}')

# 保存模型
print('保存模型')
torch.save(model.state_dict(), 'credit_score_model.pth')
print('done')
