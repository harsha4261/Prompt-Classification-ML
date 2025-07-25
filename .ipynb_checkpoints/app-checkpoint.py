import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Download all required NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Open Multilingual Wordnet

# Load models and encoders
@st.cache_resource
def load_models():
    vectorizer = joblib.load("models/ML Models/tfidf_vectorizer.pkl")
    cluster_model = joblib.load("models/ML Models/cluster_LR_model.pkl")
    subclass_model = joblib.load("models/ML Models/subclass_LR_model.pkl")
    cluster_encoder = joblib.load("models/ML Models/cluster_label_encoder.pkl")
    subclass_encoder = joblib.load("models/ML Models/subclass_label_encoder.pkl")
    return vectorizer, cluster_model, subclass_model, cluster_encoder, subclass_encoder

vectorizer, cluster_model, subclass_model, cluster_encoder, subclass_encoder = load_models()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def predict_cluster_and_subclass(new_prompt):
    processed_prompt = preprocess_text(new_prompt)
    X_new = vectorizer.transform([processed_prompt])
    cluster_probabilities = cluster_model.predict_proba(X_new)
    subclass_probabilities = subclass_model.predict_proba(X_new)
    cluster_predicted_index = cluster_probabilities.argmax(axis=1)
    subclass_predicted_index = subclass_probabilities.argmax(axis=1)
    cluster_confidence_score = cluster_probabilities[0][cluster_predicted_index][0]
    subclass_confidence_score = subclass_probabilities[0][subclass_predicted_index][0]
    predicted_cluster_label = cluster_encoder.inverse_transform(cluster_predicted_index)[0]
    predicted_subclass_label = subclass_encoder.inverse_transform(subclass_predicted_index)[0]
    return predicted_cluster_label, cluster_confidence_score, predicted_subclass_label, subclass_confidence_score

# Streamlit UI
st.title("Prompt Classification Web App")
st.write("Enter a prompt below to classify it into a cluster and subclass.")

user_prompt = st.text_area("Enter your prompt:", "How to use OpenAI's API within Streamlit?")

if st.button("Classify Prompt"):
    if user_prompt.strip():
        cluster, cluster_conf, subclass, subclass_conf = predict_cluster_and_subclass(user_prompt)
        st.success(f"**Cluster:** {cluster} (Confidence: {cluster_conf:.2f})")
        st.success(f"**Subclass:** {subclass} (Confidence: {subclass_conf:.2f})")
    else:
        st.warning("Please enter a prompt to classify.") 