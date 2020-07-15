# classifier/application/routes.py
from flask import Flask,request,jsonify
from application import app
from classificator import SpamClassificator

rus_classificator = SpamClassificator(dataset_language = 'ru')
rus_classificator.train()
en_classificator = SpamClassificator(dataset_language = 'en')
en_classificator.train()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/hello_user', methods=['POST'])
def hello_user():
    data = request.json
    user = data['user']
    if user is None:
        params = ', '.join(data.keys()) 
        return jsonify({'message': f'Parametr "{params}" is invalid'}), 400 
    else:
        return f'hello {user}'

@app.route('/increase_number', methods=['POST'])
def increase_number():
    data = request.json
    number = data['number']
    if number is None:
        params = ', '.join(data.keys()) 
        return jsonify({'message': f'Parametr "{params}" is invalid'}), 400 
    else:
        return f'{number+1}'

#рут для русского датасета
@app.route('/classify_text_ru', methods=['POST'])
def classify_text_ru():
    data = request.json
    text = data.get('text') 
    if text is None:
        params = ', '.join(data.keys()) 
        return jsonify({'message': f'Parametr "{params}" is invalid'}), 400 
    else:
        result = rus_classificator.classify(text)
        return jsonify({'result': result})

#рут для английского датасета
@app.route('/classify_text_en', methods=['POST'])
def classify_text_en():
    data = request.json
    text = data.get('text') 
    if text is None:
        params = ', '.join(data.keys()) 
        return jsonify({'message': f'Parametr "{params}" is invalid'}), 400 
    else:
        result = en_classificator.classify(text)
        return jsonify({'result': result})

