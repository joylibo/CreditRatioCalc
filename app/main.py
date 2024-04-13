from fastapi import FastAPI, Request, HTTPException
from typing import List
from transformers import BertTokenizer, BertModel
from fastapi.templating import Jinja2Templates
import logging
from sqlmodel import Session, select
from database import engine
from models import (
    ResidentCreditScore,
    CreditWarningRequest,
    CreditWarningResponse,
    CreditPredictionByGroupRequest, 
    CreditPredictionByGroupResponse,
    CreditPredictionRequest,
    CreditPredictionResponse,
    SceneName,
    ScenePerceptionResponse,
    CurrentSceneRequest,
    TextBatch,
    SimilarityScore,
    Weights
)
app = FastAPI()



# 设置日志配置
logging.basicConfig(filename='api_logs.log', level=logging.INFO)




@app.get("/get_score")
def get_score():
    """一个测试接口，用来测试数据库的连接
    """
    with Session(engine) as session:
        # 查询前10行数据
        query = select(ResidentCreditScore).limit(5)
        results = session.exec(query)
        return results.all()



