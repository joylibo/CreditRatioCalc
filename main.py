from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import euclidean
import torch
from fastapi.templating import Jinja2Templates

app = FastAPI()

# 指定本地模型和分词器的路径
local_model_path = './bert-base-chinese'

# 从本地加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)

class Weights(BaseModel):
    cosine: float = 1.0
    euclidean: float = 0.0
    manhattan: float = 0.0

class TextBatch(BaseModel):
    reference_text: str
    texts_to_compare: List[str]
    weights: Weights

class SimilarityScore(BaseModel):
    similarity: float
    score: float


templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("similarity-form.html", {"request": request})

@app.post("/get_bulk_similarity/")
def get_bulk_similarity(text_batch: TextBatch) -> List[SimilarityScore]:
    """Get similarity between some text"""
    # 实现计算相似度的逻辑
    reference_text = text_batch.reference_text
    texts_to_compare = text_batch.texts_to_compare
    weights = text_batch.weights

    reference_encoded = tokenizer(reference_text, return_tensors='pt')
    with torch.no_grad():
        reference_output = model(**reference_encoded)
    reference_embedding = reference_output.last_hidden_state[:, 0, :]

    ## similarity_scores = List[SimilarityScore]
    similarities = []
    total_weight = sum([weight for weight in weights.__dict__.values()])

    for text in texts_to_compare:
        encoded_input = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embedding = model_output.last_hidden_state[:, 0, :]
        
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

        final_score = sum(individual_similarity)
        similarities.append(final_score.item())

    min_val = min(similarities)
    max_val = max(similarities)

    # 计算每个元素的分数
    scores = [60 + (x - min_val) * (100 - 60) / (max_val - min_val) for x in similarities]

    return [SimilarityScore(similarity=s, score=scores[i]) for i, s in enumerate(similarities)]
