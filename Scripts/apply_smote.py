import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Create necessary directories
os.makedirs("models/smote", exist_ok=True)
os.makedirs("results/smote", exist_ok=True)

# ✅ Load Vectorized Data
train_df = pd.read_csv('./data/preprocessed_train.csv')
vectorizer = joblib.load('models/vectorize/tfidf_vectorizer.pkl')
X_train_tfidf = vectorizer.transform(train_df['cleaned_review_text'])
y_train = train_df['sentiment']

# ✅ Analyze Class Distribution Before SMOTE
class_distribution_before = y_train.value_counts()
with open("results/smote/class_distribution_before.txt", "w", encoding="utf-8") as f:
    f.write(f"Class Distribution Before SMOTE:\n{class_distribution_before}\n")

# ✅ Visualize Class Distribution Before SMOTE
plt.figure(figsize=(6, 4))
sns.barplot(x=class_distribution_before.index, y=class_distribution_before.values, palette="viridis")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Class Distribution Before SMOTE")
plt.savefig("results/smote/class_distribution_before.png")
plt.close()

# ✅ Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

# ✅ Verify Class Distribution After SMOTE
class_distribution_after = pd.Series(y_resampled).value_counts()
with open("results/smote/class_distribution_after.txt", "w", encoding="utf-8") as f:
    f.write(f"Class Distribution After SMOTE:\n{class_distribution_after}\n")

# ✅ Visualize Class Distribution After SMOTE
plt.figure(figsize=(6, 4))
sns.barplot(x=class_distribution_after.index, y=class_distribution_after.values, palette="coolwarm")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Class Distribution After SMOTE")
plt.savefig("results/smote/class_distribution_after.png")
plt.close()

# ✅ Save the Balanced Dataset
balanced_train_df = pd.DataFrame(X_resampled.toarray(), columns=vectorizer.get_feature_names_out())
balanced_train_df['sentiment'] = y_resampled
balanced_train_df.to_csv("data/balanced_train.csv", index=False)

# ✅ Save SMOTE Model
joblib.dump(smote, 'models/smote/smote_model.pkl')

print("✅ SMOTE applied successfully. Results saved in 'results/smote/' and 'models/smote/'.")
