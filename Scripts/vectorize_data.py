import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer

# Create necessary directories
os.makedirs("models/vectorize", exist_ok=True)
os.makedirs("results/vectorize", exist_ok=True)

# ✅ Load preprocessed datasets
train_df = pd.read_csv('./data/preprocessed_train.csv')
test_df = pd.read_csv('./data/preprocessed_test.csv')

# ✅ TF-IDF Vectorization for Naïve Bayes
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to 5000 for efficiency
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['cleaned_review_text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['cleaned_review_text'])

# ✅ Save Vectorizer
joblib.dump(tfidf_vectorizer, 'models/vectorize/tfidf_vectorizer.pkl')

# ✅ Save TF-IDF Summary
tfidf_summary = f"Number of Features: {len(tfidf_vectorizer.get_feature_names_out())}\n\nSample Features:\n{tfidf_vectorizer.get_feature_names_out()[:50]}\n"
with open("results/vectorize/tfidf_summary.txt", "w", encoding="utf-8") as f:
    f.write(tfidf_summary)

# ✅ Save Sample Vectorized Data
vectorized_output = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()).head()
vectorized_output.to_csv("results/vectorize/tfidf_vectorized_output.csv", index=False, encoding="utf-8")

# ✅ Tokenization for BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(text):
    return bert_tokenizer(text, truncation=True, padding="max_length", max_length=128)

train_df['tokenized'] = train_df['cleaned_review_text'].apply(lambda x: tokenize_text(x))
test_df['tokenized'] = test_df['cleaned_review_text'].apply(lambda x: tokenize_text(x))

# ✅ Save Sample Tokenized Data for BERT
with open("results/vectorize/bert_tokenized_output.txt", "w", encoding="utf-8") as f:
    f.write(str(train_df[['cleaned_review_text', 'tokenized']].head()))

# ✅ Save Tokenizer
bert_tokenizer.save_pretrained("models/vectorize/bert_tokenizer")

print("✅ Vectorization completed successfully. Results saved in 'models/vectorize/' and 'results/vectorize/'.")
