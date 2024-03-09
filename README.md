# CreditRatioCalc

依据文本相似度实现的信用评估模型

## 部署说明
基础依赖：

* Python 3.11.8
* Flask 2.2.5
* numpy==1.26.4
* scikit-learn==1.4.1.post1
* scipy==1.12.0
* transformers

## WSGI HTTP服务器
gunicorn==21.2.0

## 外部依赖
代码 `CreditRatioCalc/app/routes.py` 写了如下依赖
```
# 指定本地模型和分词器的路径
local_model_path = '../bert-base-chinese'
```
因此请确保本地存在bert-base-chinese模型，并把它存放在 `CreditRatioCalc`目录下

其地址在 https://huggingface.co/google-bert/bert-base-chinese
需要自行下载全部模型文件，由于比较大，git仓库中没有这些文件


## 测试你的环境是否满足项目运行要求
```
cd CreditRatioCalc
python run.py
```
如果返回如下信息，说明运行成功
```
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 106-051-265

```

## 程序启动命令

```
nohup gunicorn -w 4 -b 0.0.0.0:8000 run:app &
```

在Gunicorn中，-w参数用来指定worker进程的数量。worker进程负责处理传入的HTTP请求。通常来说，可以将worker进程数量设置为服务器的CPU核心数的2到4倍，以充分利用服务器资源。

如果服务器的内存很低，可以考虑减少worker进程的数量，以降低内存的使用量。最小的worker进程数量取决于你的应用程序的负载和性能需求。通常来说，可以将worker进程数量设置为1或2，以确保服务器仍然可以处理请求。

你可以根据服务器的实际情况逐步调整worker进程数量，并监控服务器的性能和资源使用情况，以找到最适合你的应用程序的worker进程数量。

## 程序停止命令
```
pkill gunicorn
```
