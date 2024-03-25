from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel,Field, constr, validator
from typing import List
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import euclidean
import torch
from fastapi.templating import Jinja2Templates
import logging
from enum import Enum
from typing import Dict
from datetime import datetime

app = FastAPI()

# 指定本地模型和分词器的路径
local_model_path = './bert-base-chinese'

# 从本地加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)

# 设置日志配置
logging.basicConfig(filename='api_logs.log', level=logging.INFO)

class Weights(BaseModel):
    cosine: float = 1.0
    euclidean: float = 0.0
    manhattan: float = 0.0

class TextBatch(BaseModel):
    reference_text: str = Field(
        default=None, title="计算相似度的目标文本", max_length=200
    )
    texts_to_compare: List[str] = Field(
        examples=[["str1", "str2"]],  
        title="待检测文本列表",
        description="当只有一个参数的时候，无论相似度是多少，占比都会是100%",
    )
    weights: Weights

class SimilarityScore(BaseModel):
    similarity: float
    rates: float

# 定义势态感知的request结构
class SceneName(str, Enum):
    SHE_QU = "平安社区"
    DANG_JIAN = "信用党建"
    YANG_LAO = "信用养老"

class CurrentSceneRequest(BaseModel):
    scene_name: SceneName
    perception_deadline: datetime

# 定义势态感知的返回结构
class ScenePerceptionResponse(BaseModel):
    current_scene: Dict[str, str]
    scene_perception: Dict[str, str]

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    """
    请求root的时候，会向用户发送一个页面，页面上包含一个文本框，用户可以输入文本，点击提交按钮后，会将文本发送给后端，后端会返回一个相似度分数。
    """
    return templates.TemplateResponse("similarity-form.html", {"request": request})

@app.post("/perception")
def perception_handler(request_data: CurrentSceneRequest):
    if request_data.scene_name == SceneName.SHE_QU:
        response_data = ScenePerceptionResponse(
            current_scene={},
            scene_perception={
                "吸毒人员": -0.10,
                "刑满释放人员": 0,
                "精神病人员": 0,
                "低保人员": 0,
                "社区矫正": 0,
                "空巢老人": 0,
                "五保人员": 0.10,
                "残疾人": 0.12,
                "重大疾病致困人员": 0,
                "邪教人员": 0
            }
        )
    elif request_data.scene_name == SceneName.DANG_JIAN:
        response_data = ScenePerceptionResponse(
            current_scene={},
            scene_perception={
                "党政机关党员": 0.10,
                "国有企业党员": 0.12,
                "民营企业党员": 0,
                "事业单位党员": 0.12,
                "社会组织党员": 0,
                "两新组织党员": 0.18,
                "村社基层组织党员": 0,
                "其他": 0
            }
        )
    elif request_data.scene_name == SceneName.YANG_LAO:
        response_data = ScenePerceptionResponse(
            current_scene={},
            scene_perception={
                "独居老人": 0,
                "高龄老人": 0,
                "伤残老人": 0.01,
                "困难老人": 0,
                "社区养老服务人员": 0
            }
        )
    return response_data


@app.post("/get_bulk_similarity/")
def get_bulk_similarity(text_batch: TextBatch) -> List[SimilarityScore]:
    """Get similarity between reference text and multiple texts
        
        接收一个目标文本和一批待检测文本,返回每个待检测文本与检测文本的相似度，以及一个横向比较的百分比值
    """
    reference_text = text_batch.reference_text
    texts_to_compare = text_batch.texts_to_compare
    weights = text_batch.weights

    # 记录请求时间和参数
    log_message = f"Request received at {datetime.now()} with the following parameters: reference_text={reference_text}, texts_to_compare={texts_to_compare}, weights={weights}"
    logging.info(log_message)

    reference_embedding = get_embedding(reference_text)

    similarities = []
    total_weight = sum([weight for weight in weights.__dict__.values()])

    for text in texts_to_compare:
        sentence_embedding = get_embedding(text)
        individual_similarity = calculate_individual_similarity(reference_embedding, sentence_embedding, weights, total_weight)
        final_score = sum(individual_similarity)
        similarities.append(final_score.item())

    rates = calculate_scores(similarities)

    return [SimilarityScore(similarity=s, rates=rates[i]) for i, s in enumerate(similarities)]

def get_embedding(text: str) -> torch.Tensor:
    """Get the BERT embedding of a text"""
    encoded_input = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embedding = model_output.last_hidden_state[:, 0, :]
    return sentence_embedding

def calculate_individual_similarity(reference_embedding: torch.Tensor, sentence_embedding: torch.Tensor, weights: Weights, total_weight: float) -> List[torch.Tensor]:
    """Calculate individual similarities based on different metrics"""
    individual_similarity = []

    if weights.cosine > 0:
        cosine_sim = torch.nn.functional.cosine_similarity(reference_embedding, sentence_embedding, dim=1)
        individual_similarity.append(cosine_sim * (weights.cosine / total_weight))

    if weights.euclidean > 0:
        euclidean_dist = torch.norm(reference_embedding - sentence_embedding, p=2, dim=1)
        euclidean_sim = (1 / (1 + euclidean_dist)) * (weights.euclidean / total_weight)
        individual_similarity.append(euclidean_sim)

    if weights.manhattan > 0:
        manhattan_dist = torch.norm(reference_embedding - sentence_embedding, p=1, dim=1)
        manhattan_sim = (1 / (1 + manhattan_dist)) * (weights.manhattan / total_weight)
        individual_similarity.append(manhattan_sim)

    return individual_similarity

def calculate_scores(similarities: List[float]) -> List[float]:
    """Calculate normalized scores on the list of similarities"""
    sum_of_similarities = sum(similarities)
    rates = [x / sum_of_similarities for x in similarities]

    # 四舍五入并保留4位小数
    rounded_rates = [round(rate, 4) for rate in rates]

    # 计算修正后的总和
    rounded_sum = sum(rounded_rates)

    # 对最后一个元素进行修正，使总和等于1
    rounded_rates[-1] += 1 - rounded_sum

    # 返回修正后的百分比列表
    return rounded_rates