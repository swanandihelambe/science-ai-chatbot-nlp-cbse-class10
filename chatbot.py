import json
import nltk
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Downloads (minimal)
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load intents
with open("cbse_class10_intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data["intents"]:
    for question in intent["utterances"]:
        sentences.append(question)
        labels.append(intent["intent"])

processed_sentences = [preprocess_text(s) for s in sentences]

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_sentences)

# Model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X, labels)

# Save model & vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(classifier, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ CBSE Class 10 Science Chatbot (Bio + Chem + Physics) is ready!")

# Chat loop
while True:
    user_input = input("\nAsk a CBSE Class 10 science question (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    cleaned = preprocess_text(user_input)
    vector = vectorizer.transform([cleaned])

    # Prediction + confidence
    probabilities = classifier.predict_proba(vector)
    confidence = max(probabilities[0])
    predicted_intent = classifier.predict(vector)[0]

    # DEBUG 
    #print(f"DEBUG → Intent: {predicted_intent}, Confidence: {confidence:.2f}")

    # threshold with respect to intents
    if confidence < 0.15:
        print("Bot: I'm not confident about this. Please ask a CBSE Class 10 science question.")
        continue

    # Fetch response
    response = "Sorry, I can answer only CBSE Class 10 science questions."

    for intent in data["intents"]:
        if intent["intent"] == predicted_intent:
            response = (
                f"📘 Subject: {intent['subject']}\n"
                f"📗 Chapter: {intent['chapter']}\n"
                f"💡 Answer: {intent['response']}\n"
                #f"📊 Confidence: {confidence:.2f}"
            )
            break

    print("\nBot:\n", response)
