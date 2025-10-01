# CreditRatioCalc

依据文本相似度实现的信用评估模型

## 部署说明
基础依赖：

请参考项目根目录下的 `requirements.txt` 文件安装所有必要的Python包：

```bash
pip install -r requirements.txt
```

主要依赖包括：
* fastapi>=0.68.0
* uvicorn[standard]>=0.15.0
* sqlmodel>=0.0.6
* pandas>=1.3.0
* numpy>=1.21.0
* torch>=1.9.0
* torchvision>=0.10.0
* scikit-learn>=1.0.0
* joblib>=1.0.0
* tqdm>=4.62.0
* transformers>=4.12.0
* pydantic>=1.8.0
* jinja2>=3.0.0
* openpyxl>=3.0.0
* openai>=1.0.0
* python-dotenv>=0.19.0
* pymysql>=1.0.0

## ASGI HTTP服务器
uvicorn

## 外部依赖
代码 `CreditRatioCalc/app/routes/similarity.py` 写了如下依赖
```
# 构建本地模型的路径
local_model_path = os.path.join(current_dir, '..', '..', 'bert-base-chinese')
```
因此请确保本地存在bert-base-chinese模型，并把它存放在 `CreditRatioCalc`目录下, 与`app`目录同级

其地址在 https://huggingface.co/google-bert/bert-base-chinese
需要自行下载全部模型文件，由于比较大，git仓库中没有这些文件


## 环境配置
需要关注`app/config/credit_db_conf.py`文件中的配置,这个文件中保存了数据库的配置，所以没有放入git仓库，部署时候需要手动创建,示例
```
DATABASE_CONFIG = {
    'username': 'xxxx',
    'password': 'xxxx',
    'hostname': 'xxx.xxx.xxx.xxx',
    'database': 'xxxx',
    'port': ***
}
```

## 测试你的环境是否满足项目运行要求


## 程序启动命令

确保首先切换到项目根目录（包含app目录的目录），然后运行以下命令：

调试模式
```
cd {PATH_TO_CreditRatioCalc}
uvicorn app.main:app --reload
```

生产模式
```
cd {PATH_TO_CreditRatioCalc}
nohup uvicorn app.main:app &
```

## 程序停止命令
```
pkill uvicorn
```

## 定时任务配置

项目需要配置定时任务来每日执行 `daily_forecast_all.py` 脚本进行信用分数预测。推荐在每天凌晨2点执行。

### 配置定时任务的步骤：

1. 首先确认conda环境的路径：
   ```bash
   which conda
   ```
   这将输出conda可执行文件的完整路径。

2. 查找脚本文件的路径：
   ```bash
   find / -name "daily_forecast_all.py" 2>/dev/null
   ```
   这将输出脚本的完整路径。

3. 编辑crontab配置：
   ```bash
   crontab -e
   ```

4. 在crontab文件中添加以下行（请将CONDA_PATH和SCRIPT_PATH替换为实际路径）：
   ```
   0 2 * * * /bin/bash -c "source CONDA_PATH activate py3118 && nohup python SCRIPT_PATH" > SCRIPT_DIR/daily_forecast_all.log 2>&1 &
   ```

   例如，如果`which conda`输出为`/home/user/miniconda3/bin/conda`，`find`命令找到脚本在`/path/to/CreditRatioCalc/daily_forecast_all.py`，则配置为：
   ```
   0 2 * * * /bin/bash -c "source /home/user/miniconda3/bin/activate py3118 && nohup python /path/to/CreditRatioCalc/daily_forecast_all.py" > /path/to/CreditRatioCalc/daily_forecast_all.log 2>&1 &
   ```

5. 保存并退出编辑器。

6. 验证crontab配置：
   ```bash
   crontab -l
   ```
