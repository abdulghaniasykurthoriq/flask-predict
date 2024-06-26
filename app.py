import os
from flask import Flask, request, jsonify
import joblib
import re
import requests

app = Flask(__name__)

# Load the trained model and vectorizer
model_filename = 'hoax_detector_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

model = joblib.load(model_filename)
vectorizer = joblib.load(vectorizer_filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return jsonify({'prediction': prediction[0]})

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words with length 1 or 2
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
