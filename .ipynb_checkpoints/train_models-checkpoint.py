import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Download required NLTK data
print("Downloading NLTK resources...")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

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

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv("data/raw/prompts_v1.csv")
    
    # Preprocess prompts
    print("Preprocessing prompts...")
    df["processed_prompt"] = df["prompt"].apply(preprocess_text)
    
    # Encode labels
    print("Encoding labels...")
    le_cluster = LabelEncoder()
    le_sub_class = LabelEncoder()
    df["cluster_encoded"] = le_cluster.fit_transform(df["cluster"])
    df["sub_class_encoded"] = le_sub_class.fit_transform(df["sub_class"])
    
    # Feature extraction
    print("Extracting features...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["processed_prompt"])
    y_cluster = df["cluster_encoded"]
    y_sub_class = df["sub_class_encoded"]
    
    # Train cluster models
    print("\nTraining cluster models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_cluster, test_size=0.2, stratify=y_cluster)
    
    cluster_models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Naive Bayes": MultinomialNB(),
    }
    
    best_cluster_model = None
    best_cluster_acc = 0
    
    for name, model in cluster_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name} - Accuracy: {acc}, F1 Score: {f1}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        if acc > best_cluster_acc:
            best_cluster_acc = acc
            best_cluster_model = model
    
    # Train subclass models
    print("\nTraining subclass models...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_sub_class, test_size=0.2, stratify=y_sub_class)
    
    subclass_models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Naive Bayes": MultinomialNB(),
    }
    
    best_subclass_model = None
    best_subclass_acc = 0
    
    for name, model in subclass_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"{name} - Accuracy: {acc}, F1 Score: {f1}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        if acc > best_subclass_acc:
            best_subclass_acc = acc
            best_subclass_model = model
    
    # Save models and encoders
    print("\nSaving models and encoders...")
    joblib.dump(best_cluster_model, "models/ML Models/cluster_LR_model.pkl")
    joblib.dump(best_subclass_model, "models/ML Models/subclass_LR_model.pkl")
    joblib.dump(vectorizer, "models/ML Models/tfidf_vectorizer.pkl")
    joblib.dump(le_cluster, "models/ML Models/cluster_label_encoder.pkl")
    joblib.dump(le_sub_class, "models/ML Models/subclass_label_encoder.pkl")
    
    print("\nTraining complete!")
    print(f"Best cluster model accuracy: {best_cluster_acc}")
    print(f"Best subclass model accuracy: {best_subclass_acc}")

if __name__ == "__main__":
    main() 