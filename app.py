from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)
CORS(app)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model_save_path = "./saved_model"
tokenizer_save_path = "./saved_tokenizer"

model = BertForSequenceClassification.from_pretrained(model_save_path).to(device)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_save_path)

sentiment_mapping = {0: 'neutral', 1: 'negative', 2: 'positive'}

# predict the sentiment of the text
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    # necessary to call .cpu() to change back to using cpu
    # https://stackoverflow.com/questions/53467215/convert-pytorch-cuda-tensor-to-numpy-array 
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    return sentiment_mapping[predictions[0]]

# predicting the text
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    sentiment = predict_sentiment(text)
    return jsonify({'sentiment': sentiment})

# scraping data from Yahoo Finance
@app.route('/scrape', methods=['GET'])
def scrape():
    url = "https://finance.yahoo.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    headlines = []
    h3_tags = soup.find_all('h3', class_='clamp yf-1044anq')
    for h3 in h3_tags:
        headline = h3.get_text(strip=True)
        if headline: 
            sentiment = predict_sentiment(headline)
            headlines.append({'headline': headline, 'sentiment': sentiment})

    return jsonify({'headlines': headlines})

if __name__ == '__main__':
    app.run(debug=True)