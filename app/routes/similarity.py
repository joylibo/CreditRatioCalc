from fastapi import FastAPI
from typing import List
from transformers import BertTokenizer, BertModel
import torch
import logging
from datetime import datetime
from app.models.text_batch import TextBatch, SimilarityScore, Weights

app = FastAPI()

# 设置日志配置
logging.basicConfig(filename='api_logs.log', level=logging.INFO)

# 指定本地模型和分词器的路径
local_model_path = '../bert-base-chinese'

# 从本地加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)

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
