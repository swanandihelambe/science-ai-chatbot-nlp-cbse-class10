# 🧠 CBSE Class 10 Science AI Chatbot

An AI-powered chatbot designed to answer **CBSE Class 10 Science definitions** using Natural Language Processing (NLP).

This project was built to provide **clear, exam-oriented, short-step answers** for Biology, Chemistry, and Physics — something I personally struggled to find during my Class 10 board preparation.

---

## 🚀 Features
- 📘 Supports **Biology, Chemistry & Physics**
- 🧠 NLP-based **intent classification**
- 🔎 TF-IDF + Logistic Regression model
- 📊 Confidence-based fallback handling
- 🎨 Streamlit UI with:
  - Light / Dark / Gradient / Solid modes
  - 17 customizable color themes

---

## 🛠️ Tech Stack
- **Python**
- **NLTK** (text preprocessing)
- **Scikit-learn** (TF-IDF + Logistic Regression)
- **Streamlit** (UI)

---

## ⚙️ How it Works
1. Created a structured **intents dataset** in JSON
2. Preprocessed text (lowercasing, stopword removal, lemmatization)
3. Converted text into vectors using **TF-IDF**
4. Trained an **intent classification model**
5. Built an interactive UI using **Streamlit**

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
