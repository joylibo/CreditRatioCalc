from flask import request, jsonify, render_template
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import euclidean
import torch
from app import app

# 指定本地模型和分词器的路径
local_model_path = './bert-base-chinese'

# 从本地加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)

@app.route('/')
def index():
    return render_template('index.html')

    

@app.route('/concat_two', methods=['POST'])
def concat_two():
    data = request.json
    str1 = data.get('str1', '')
    str2 = data.get('str2', '')
    return jsonify({'result': str1 + str2})

@app.route('/concat_multiple', methods=['POST'])
def concat_multiple():
    data = request.json
    strings = data.get('strings', [])
    result = ''.join(strings)
    return jsonify({'result': result})

@app.route('/average', methods=['POST'])
def average():
    data = request.json
    numbers = data.get('numbers', [])
    if numbers and isinstance(numbers, list):
        avg = sum(numbers) / len(numbers)
    else:
        avg = 0
    return jsonify({'average': avg})

@app.route('/cosine_sim_bert', methods=['POST'])
def cosine_sim_bert():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    if text1 and text2:
        # 使用分词器处理文本
        encoded_input1 = tokenizer(text1, return_tensors='pt')
        encoded_input2 = tokenizer(text2, return_tensors='pt')

        # 使用模型生成文本嵌入
        with torch.no_grad():
            model_output1 = model(**encoded_input1)
            model_output2 = model(**encoded_input2)

        # 获取最后一层隐藏状态的第一个token（CLS token）作为句子的表示
        sentence_embedding1 = model_output1.last_hidden_state[:, 0, :]
        sentence_embedding2 = model_output2.last_hidden_state[:, 0, :]
    
        # 计算两个句子嵌入的余弦相似度
        cosine_similarity = torch.nn.functional.cosine_similarity(sentence_embedding1, sentence_embedding2)
        result = cosine_similarity.item()
    else:
        result = 0
    return jsonify({'result': result})

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    if text1 and text2:
        # 使用分词器处理文本
        encoded_input1 = tokenizer(text1, return_tensors='pt')
        encoded_input2 = tokenizer(text2, return_tensors='pt')

        # 使用模型生成文本嵌入
        with torch.no_grad():
            model_output1 = model(**encoded_input1)
            model_output2 = model(**encoded_input2)

        # 获取最后一层隐藏状态的第一个token（CLS token）作为句子的表示
        sentence_embedding1 = model_output1.last_hidden_state[:, 0, :]
        sentence_embedding2 = model_output2.last_hidden_state[:, 0, :]
    
        # 计算两个句子嵌入的余弦相似度并归一化
        cosine_similarity = torch.nn.functional.cosine_similarity(sentence_embedding1, sentence_embedding2)
        normalized_cosine_similarity = (cosine_similarity.item() + 1) / 2
        
        # 计算两个句子嵌入的欧氏距离并归一化
        euclidean_distance = euclidean(sentence_embedding1.squeeze().numpy(), sentence_embedding2.squeeze().numpy())
        normalized_euclidean_distance = 1 / (1 + euclidean_distance)
        
        # 综合考虑余弦相似度和欧氏距离，这里使用简单的平均值
        similarity_score = (normalized_cosine_similarity + normalized_euclidean_distance) / 2
        
        result = similarity_score
    else:
        result = 0
    return jsonify({'result': result})

@app.route('/bulk_similarity', methods=['POST'])
def bulk_similarity():
    try:
        data = request.json
        reference_text = data.get('reference_text', '')
        texts_to_compare = data.get('texts_to_compare', [])

        if not reference_text or not texts_to_compare:
            return jsonify({'error': 'Reference text and texts to compare are required.'}), 400

        reference_encoded = tokenizer(reference_text, return_tensors='pt')
        with torch.no_grad():
            reference_output = model(**reference_encoded)
        reference_embedding = reference_output.last_hidden_state[:, 0, :]

        similarities = []
        for text in texts_to_compare:
            encoded_input = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embedding = model_output.last_hidden_state[:, 0, :]

            # 计算余弦相似度并归一化
            cosine_similarity = torch.nn.functional.cosine_similarity(reference_embedding, sentence_embedding, dim=1)
            normalized_cosine_similarity = (cosine_similarity + 1) / 2

            # 计算欧氏距离并归一化
            euclidean_distance = euclidean(reference_embedding.squeeze().numpy(), sentence_embedding.squeeze().numpy())
            normalized_euclidean_distance = 1 / (1 + euclidean_distance)

            # 计算综合相似度得分
            similarity_score = (normalized_cosine_similarity + normalized_euclidean_distance) / 2

            # 确保将Tensor转换为Python的浮点数
            similarity_score_value = similarity_score.item()  # 如果similarity_score是一个单元素Tensor

            similarities.append(similarity_score_value)

        return jsonify({'similarities': similarities})
    except Exception as e:
        # 捕获任何异常并返回错误信息
        return jsonify({'error': 'An error occurred during processing: ' + str(e)}), 500