import re
import string
import joblib
import pandas as pd

model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
print("✅ Model and vectorizer loaded successfully!")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def predict_review(review_text):
    clean_review = clean_text(review_text)
    vect_review = vectorizer.transform([clean_review])
    prediction = model.predict(vect_review)[0]
    label = "Fake Review ❌" if prediction == 1 else "Real Review ✅"
    return label
while True:
    print("\nEnter a review to test (or type 'exit' to quit):")
    user_input = input()
    if user_input.lower() == 'exit':
        break
    result = predict_review(user_input)
    print(f"Prediction: {result}")
