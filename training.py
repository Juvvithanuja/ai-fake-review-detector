import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
df = pd.read_csv("fake_review.csv")
print("Dataset Loaded Successfully!")
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
df['clean_review'] = df['review_text'].apply(clean_text)
df['label'] = df['label'].map({'fake': 1, 'real': 0}).fillna(df['label'])
df = df.dropna(subset=['clean_review', 'label'])
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['label'], test_size=0.3, random_state=42
)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("\n✅ Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
joblib.dump(model, "fake_review_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n🎉 Model and vectorizer saved successfully!")

