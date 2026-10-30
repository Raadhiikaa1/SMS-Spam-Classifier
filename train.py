import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load dataset
df = pd.read_csv(
    "data/SMSSpamCollection.csv",
    encoding="latin-1"
)

# 2. Keep only the required columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# 3. Convert labels to numbers
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# 5. Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 6. Train the machine-learning model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

# 7. Test the model
y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))

# 8. Save the model
with open("spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

# 9. Save the vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel and vectorizer saved successfully!")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Spam Message Classifier - Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()