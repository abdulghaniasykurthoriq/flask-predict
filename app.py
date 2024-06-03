from flask import Flask, request, jsonify
import joblib
import re
import requests
import os

app = Flask(__name__)

# URLs to the model and vectorizer
# model_url = 'https://storage.googleapis.com/bucket-flask-expres/hoax_detector_model.pkl'
# vectorizer_url = 'https://storage.googleapis.com/bucket-flask-expres/tfidf_vectorizer.pkl'

# Local filenames
model_filename = 'hoax_detector_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

# Function to download files
def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()  # Check if the download is successful
    with open(filename, 'wb') as f:
        f.write(response.content)

# Download the model and vectorizer if not already downloaded
if not os.path.exists(model_filename):
    download_file(model_url, model_filename)

if not os.path.exists(vectorizer_filename):
    download_file(vectorizer_url, vectorizer_filename)

# Load the trained model and vectorizer
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
    app.run(debug=True)
