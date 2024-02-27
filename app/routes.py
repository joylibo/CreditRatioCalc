from flask import request, jsonify
from app import app

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
