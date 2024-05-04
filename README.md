# CreditRatioCalc

依据文本相似度实现的信用评估模型

## 部署说明
基础依赖：

* Python 3.11.8
* fastAPI
* sqlmodel
* pytorch
* numpy==1.26.4
* scikit-learn==1.4.1.post1
* scipy==1.12.0
* transformers

连接数据库的工具安装：
conda install -c conda-forge sqlmodel
conda install -c conda-forge mysqlclient

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


## 测试你的环境是否满足项目运行要求


## 程序启动命令

调试模式
```
uvicorn app.main:app --reload
```    

生产模式
```
nohup uvicorn app.main:app &
```

## 程序停止命令
```
pkill uvicorn
```
