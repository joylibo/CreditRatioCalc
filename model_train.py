import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


print("lgb版本：" + lgb.__version__)

# 读取数据
print('读取数据')
data = pd.read_excel('/Users/libo/Downloads/历史信用及行为数据.xlsx')
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

# 构造滑动窗口数据
print('构造滑动窗口数据')
X, y = [], []
window_size = 100
future_days = 30
for _, group in tqdm(data.groupby('resident_id'), desc='Processing data'):
    for i in range(window_size, len(group) - future_days + 1):
        X.append(group.iloc[i-window_size:i, 2:].values.reshape(-1))
        y.append(group.iloc[i:i+future_days, 2].values.flatten())
print('done')

# 划分训练集和测试集
print('划分训练集和测试集')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('done')

# 创建LightGBM数据集
print('创建LightGBM数据集')
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
print('done')

# 设置参数
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 训练模型
print('训练模型')
gbm = lgb.train(params,
                train_data,
                num_boost_round=2000,
                valid_sets=[train_data, test_data],
                valid_names=['train', 'valid'],
                early_stopping_rounds=100,
                verbose_eval=10)  # 每10轮输出一次信息
print('done')

# 模型评估
print('模型评估')
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')
print('done')

# 保存模型
print('保存模型')
gbm.save_model('credit_score_model.txt')
print('done')