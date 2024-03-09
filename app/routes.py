from flask import request, jsonify
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
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>文本相似度计算</title>
    </head>
    <body>
        <h2>输入两段文本以计算余弦相似度</h2>
        <form id="similarityForm">
            <div>
                <label for="text1">文本1:</label><br>
                <textarea id="text1" name="text1" rows="4" cols="50"></textarea>
            </div>
            <div>
                <label for="text2">文本2:</label><br>
                <textarea id="text2" name="text2" rows="4" cols="50"></textarea>
            </div>
            <input type="button" value="计算相似度" onclick="calculateSimilarity()">
        </form>
        <div id="result"></div>

        <script>
        function calculateSimilarity() {
            var text1 = document.getElementById("text1").value;
            var text2 = document.getElementById("text2").value;
            
            fetch('/similarity', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({text1: text1, text2: text2}),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById("result").innerHTML = "相似度得分: " + data.result;
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        }
        </script>
    </body>
    </html>
    '''

    

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
    
        # 计算两个句子嵌入的余弦相似度
        cosine_similarity = torch.nn.functional.cosine_similarity(sentence_embedding1, sentence_embedding2)
        
        # 计算两个句子嵌入的欧氏距离
        euclidean_distance = euclidean(sentence_embedding1.squeeze().numpy(), sentence_embedding2.squeeze().numpy())
        
        # 综合考虑余弦相似度和欧氏距离，这里使用简单的平均值
        similarity_score = (cosine_similarity.item() + 1 / (1 + euclidean_distance)) / 2
        
        result = similarity_score
    else:
        result = 0
    return jsonify({'result': result})

