# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re
from google.colab import files

# Step 1: Creating a simple dataset
data = {
    'text': [
        'Presiden mengumumkan program vaksinasi massal mulai minggu depan',
        'Berita palsu tentang chip mikro di vaksin COVID-19 menyebar luas',
        'Peneliti menemukan metode baru untuk mendeteksi penyakit jantung',
        'Teori konspirasi baru muncul tentang pendaratan di bulan',
        'Pemerintah meluncurkan kampanye besar untuk melawan berita palsu',
        'Orang bisa terbang dengan mengkonsumsi suplemen tertentu, ini adalah fakta!',
        'Vaksin COVID-19 telah diuji dan aman digunakan menurut WHO',
        'Berita palsu menyebutkan bahwa minum air kelapa bisa menyembuhkan COVID-19',
        'Ilmuwan menciptakan teknologi baru untuk mendaur ulang plastik',
        'Ada rumor bahwa bumi sebenarnya datar dan NASA menutupinya'
    ],
    'label': [
        'fact', 'hoax', 'fact', 'hoax', 'fact',
        'hoax', 'fact', 'hoax', 'fact', 'hoax'
    ]
}

df = pd.DataFrame(data)

# Step 2: Preprocessing the data
# Cleaning the text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words with length 1 or 2
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['text'] = df['text'].apply(preprocess_text)

# Step 3: Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Step 4: Vectorizing the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 5: Training the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Saving the model and vectorizer
joblib.dump(model, 'hoax_detector_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Downloading the model and vectorizer
files.download('hoax_detector_model.pkl')
files.download('tfidf_vectorizer.pkl')
