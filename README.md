# CreditRatioCalc

依据文本相似度实现的信用评估模型

## 部署说明
基础依赖：

* Python 3.11.8
* fastAPI
* pytorch
* numpy==1.26.4
* scikit-learn==1.4.1.post1
* scipy==1.12.0
* transformers

## ASGI HTTP服务器
uvicorn

## 外部依赖
代码 `CreditRatioCalc/main.py` 写了如下依赖
```
# 指定本地模型和分词器的路径
local_model_path = './bert-base-chinese'
```
因此请确保本地存在bert-base-chinese模型，并把它存放在 `CreditRatioCalc`目录下

其地址在 https://huggingface.co/google-bert/bert-base-chinese
需要自行下载全部模型文件，由于比较大，git仓库中没有这些文件


## 测试你的环境是否满足项目运行要求


## 程序启动命令

```
nohup uvicorn main:app --reload &
```

## 程序停止命令
```
pkill uvicorn
```
