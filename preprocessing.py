import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("fake_review.csv")  # your dataset file
print("✅ Dataset Loaded Successfully!")
print(df.head())

# 2️⃣ Basic info
print("\nDataset Info:")
print(df.info())
print("\nNull values per column:\n", df.isnull().sum())

# Clean text function
def clean_text(text):
    text = str(text).lower()                                   # lowercase
    text = re.sub(r'\[.*?\]', '', text)                        # remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)          # remove URLs
    text = re.sub(r'<.*?>+', '', text)                         # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', ' ', text)                            # remove newlines
    text = re.sub(r'\w*\d\w*', '', text)                       # remove words with numbers
    text = re.sub(r'\s+', ' ', text).strip()                   # remove extra spaces
    return text

# Apply cleaning on review_text column
df['clean_review'] = df['review_text'].apply(clean_text)

# Encode target labels
df['label'] = df['label'].map({'fake': 1, 'real': 0}).fillna(df['label'])

# Optional: drop any rows with null values in important columns
df = df.dropna(subset=['clean_review', 'label'])

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\n✅ Preprocessing Complete!")
print(f"Train samples: {X_train_tfidf.shape[0]}")
print(f"Test samples: {X_test_tfidf.shape[0]}")
