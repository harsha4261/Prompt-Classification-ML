# Basic Libraries
import pandas as pd
import numpy as np

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Text Libraries
import nltk 
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Feature Extraction Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split , cross_val_score

# Classifier Model libraries
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import tree
# from sklearn.pipeline import Pipeline

# Performance Matrix libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
import joblib
# other
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("expanded_prompts.csv")
df

print('Dataset size:',df.shape)
print('Columns are:',df.columns)
Y = df['cluster']

df.info()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# Preprocessing function
def preprocess_text(text):

    text = text.lower()
    # Removing punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenizing text
    tokens = nltk.word_tokenize(text)

    # Removing stopwords and applying lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Joining tokens back to string
    processed_text = " ".join(tokens)

    return processed_text

df["processed_prompt"] = df["prompt"].apply(preprocess_text)
df.head()

from sklearn.preprocessing import LabelEncoder

le_cluster = LabelEncoder()
le_sub_class = LabelEncoder()

df["cluster_encoded"] = le_cluster.fit_transform(df["cluster"])

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.9)
X = vectorizer.fit_transform(df["processed_prompt"])

y_cluster = df["cluster_encoded"]
# Save the cluster label encoder for later use in the Streamlit app
joblib.dump(y_cluster, "cluster_encoder.pkl")
print("Cluster encoder saved as 'cluster_encoder.pkl'")


# Save the vectorizer for later use in the Streamlit app
joblib.dump(vectorizer, "vectorizer.pkl")
print("Vectorizer saved as 'vectorizer.pkl'")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_cluster, test_size=0.2, random_state=42)

# Define models
cluster_models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),  # Enable probability for confidence scores
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
}

# Store metrics for visualization
model_metrics = {"Model": [], "Accuracy": [], "F1 Score": []}
cv_scores = {"Model": [], "CV Scores": []}
best_model = None
best_acc = 0

# Train and evaluate models
for name, model in cluster_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    model_metrics["Model"].append(name)
    model_metrics["Accuracy"].append(acc)
    model_metrics["F1 Score"].append(f1)
    if acc > best_acc:
        best_acc = acc
        best_model = model
    print(f"{name} - Accuracy: {acc}, F1 Score: {f1}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}\n")
    
    # Cross-validation scores
    cv = cross_val_score(model, X, y_cluster, cv=5, scoring="accuracy")
    cv_scores["Model"].extend([name] * len(cv))
    cv_scores["CV Scores"].extend(cv)
    
    # Store predictions for SVM
    if name == "SVM":
        svm_y_pred = y_pred
        svm_probabilities = model.predict_proba(X_test)

best_model, best_acc

joblib.dump(best_model, "cluster_LR_model.pkl")

# Visualization 2: Model Performance Comparison
plt.figure(figsize=(10, 6))
x = range(len(model_metrics["Model"]))
plt.bar([i - 0.2 for i in x], model_metrics["Accuracy"], width=0.4, label="Accuracy", color="#36A2EB")
plt.bar([i + 0.2 for i in x], model_metrics["F1 Score"], width=0.4, label="F1 Score", color="#FF6384")
plt.xticks(x, model_metrics["Model"])
plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()

# Visualization 3: Feature Importance (Random Forest)
rf_model = cluster_models["Random Forest"]
importances = rf_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()
top_indices = importances.argsort()[-10:][::-1]
top_features = [feature_names[i] for i in top_indices]
top_importances = importances[top_indices]

plt.figure(figsize=(10, 6))
plt.bar(top_features, top_importances, color="#36A2EB")
plt.title("Top 10 TF-IDF Features (Random Forest)")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Visualization 5: Cross-Validation Score Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(x="Model", y="CV Scores", data=pd.DataFrame(cv_scores), palette="Set3")
plt.title("Cross-Validation Accuracy Score Distribution")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

# Visualization 8: Prediction Confidence Histogram (SVM)
plt.figure(figsize=(10, 6))
max_probs = np.max(svm_probabilities, axis=1)
plt.hist(max_probs, bins=20, color="#36A2EB", edgecolor="black")
plt.title("Random Forest Prediction Confidence Distribution")
plt.xlabel("Max Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

cluster_LR = joblib.load("cluster_LR_model.pkl")

def predict_cluster_and_subclass(new_prompt):

    processed_prompt = preprocess_text(new_prompt)
    X_new = vectorizer.transform([processed_prompt])

    predicted_cluster = cluster_LR.predict(X_new)

    # Convert the predicted cluster from its numerical label back to the original string label
    predicted_cluster_label = le_cluster.inverse_transform(predicted_cluster)

    return predicted_cluster_label[0]

predict_cluster_and_subclass("How to use OpenAI's API within Streamlit?")

feature_names = [f"Feature {i}" for i in range(X.shape[1])]

# Plot the first decision tree from the Random Forest
plt.figure(figsize=(20,10))
plot_tree(rf_model.estimators_[0], 
          feature_names=feature_names, 
          class_names=Y.unique().astype(str).tolist(), 
          filled=True)
plt.show()

