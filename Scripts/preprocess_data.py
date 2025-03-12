import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import RandomOverSampler

# Create results directory
os.makedirs("results/preprocess", exist_ok=True)

def save_results(filename, content):
    """Save results to a text file in the results folder."""
    with open(f"results/preprocess/{filename}.txt", "w", encoding="utf-8") as file:
        file.write(content)

def preprocess_text(text):
    """Preprocess text by lowercasing, removing punctuation, URLs, numbers, and extra spaces."""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()
    return text

def extract_sentiment(rating_text):
    """Convert rating text (e.g., 'Rated 1 out of 5 stars') to sentiment label."""
    match = re.search(r'(\d) out of 5', str(rating_text))
    if match:
        rating = int(match.group(1))
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"
    return "Unknown"

def analyze_class_distribution(df, filename, title="Class Distribution"):
    """Visualize and save class distribution."""
    plt.figure(figsize=(6,4))
    df['sentiment'].value_counts().plot(kind='bar', color=['red', 'blue', 'green'])
    plt.title(title)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.savefig(f"results/preprocess/{filename}.png")
    plt.close()
    
    distribution = df['sentiment'].value_counts().to_string()
    save_results(filename, f"{title}\n\n{distribution}")

def preprocess_data(file_path):
    """Perform preprocessing, sentiment extraction, vectorization, and handling imbalanced data."""
    
    # **1️⃣ Load Dataset with Encoding Fixes**
    try:
        df = pd.read_csv(file_path, encoding="utf-8", engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="ISO-8859-1", engine="python", on_bad_lines="skip")
    
    # **2️⃣ Extract Sentiment from "Rating"**
    if 'Rating' not in df.columns:
        raise ValueError("Missing 'Rating' column in dataset.")
    
    df['sentiment'] = df['Rating'].apply(extract_sentiment)
    
    # **3️⃣ Ensure "Review Text" Exists**
    if 'Review Text' not in df.columns:
        raise ValueError("Missing 'Review Text' column in dataset.")

    # **4️⃣ Preprocessing Reviews**
    df['cleaned_review_text'] = df['Review Text'].apply(preprocess_text)
    
    # **5️⃣ Remove Unknown Sentiments**
    df = df[df['sentiment'] != "Unknown"]

    # **6️⃣ Save Overview Before and After Processing**
    save_results("Dataset_Overview_Before", df.head().to_string())
    save_results("Dataset_Overview_After", df[['Review Text', 'cleaned_review_text', 'sentiment']].head().to_string())

    # **7️⃣ Vocabulary Analysis**
    all_words = ' '.join(df['cleaned_review_text']).split()
    word_freq = Counter(all_words)
    save_results("Vocabulary_Building", f"Total unique words: {len(word_freq)}\n\nMost common words:\n{word_freq.most_common(50)}")
    
    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
    wordcloud.to_file("results/preprocess/Vocabulary_WordCloud.png")

    # **8️⃣ Split Dataset**
    df.dropna(subset=['cleaned_review_text', 'sentiment'], inplace=True)
    X = df['cleaned_review_text']
    y = df['sentiment']
    
    train, test = train_test_split(df, test_size=0.2, stratify=y, random_state=42)
    save_results("Dataset_Splitting", f"Training set: {train.shape}, Testing set: {test.shape}")

    analyze_class_distribution(train, "Class_Distribution_Before")

    # **9️⃣ Vectorization**
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(train['cleaned_review_text'])
    save_results("Vectorization", f"Vectorized Training Data Shape: {X_train_tfidf.shape}\n\nFeature Names:\n{vectorizer.get_feature_names_out()[:50]}")

    X_test_tfidf = vectorizer.transform(test['cleaned_review_text'])

    # **🔟 Handle Imbalanced Data**
    class_dist_before = train['sentiment'].value_counts().to_string()
    save_results("Class_Distribution_Before", class_dist_before)

    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(train[['cleaned_review_text']], train['sentiment'])
    
    train_resampled = pd.DataFrame({'cleaned_review_text': X_train_resampled.values.flatten(), 'sentiment': y_train_resampled})
    analyze_class_distribution(train_resampled, "Class_Distribution_After")

    # **🔹 Save Final Preprocessed Data**
    os.makedirs("data", exist_ok=True)
    train_resampled.to_csv('./data/preprocessed_train.csv', index=False)
    test[['cleaned_review_text', 'sentiment']].to_csv('./data/preprocessed_test.csv', index=False)

    print("✅ Preprocessed datasets saved successfully!")

    return train_resampled, test

if __name__ == "__main__":
    try:
        train, test = preprocess_data('./data/Amazon_Reviews_Dataset.csv')
        print("\n🚀 **Preprocessing Completed! Results saved in 'results' folder.**")
    except Exception as e:
        print(f"❌ Error: {e}")
