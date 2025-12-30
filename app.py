import streamlit as st
import json
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# NLP FUNCTIONS
@st.cache_data
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
download_nltk_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

@st.cache_data
def load_intents():
    with open("cbse_class10_intents.json", "r", encoding="utf-8") as file:
        return json.load(file)["intents"]
intents = load_intents()

def find_best_intent(user_input):
    cleaned = preprocess_text(user_input)
    best_match = None
    best_score = 0
    for intent in intents:
        intent_score = 0
        for utterance in intent["utterances"]:
            utterance_clean = preprocess_text(utterance)
            common_words = len(set(cleaned.split()) & set(utterance_clean.split()))
            intent_score = max(intent_score, common_words)
        if intent_score > best_score:
            best_score = intent_score
            best_match = intent
    return best_match, best_score / max(len(cleaned.split()), 1)

# THEME SYSTEM 
def set_theme(mode, solid_color="#6366f1"):
    css = f"""
    <style>
    .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 16px;
    }}
    /* Subject cards always black text */
    .stInfo {{
        color: black !important;
    }}
    .stInfo > div {{
        color: black !important;
    }}
    """
    
    if mode == "Dark":
        css += """
        .stApp { background-color: #0f0f23; color: #e5e7eb; }
        .stTextInput>div>div>input { background-color: #1f2937 !important; color: #f9fafb !important; border-radius: 12px !important; border: 1px solid #374151 !important; }
        section[data-testid="stSidebar"] { background-color: #1f2937; }
        """
    elif mode == "Light":
        css += """
        .stApp { background-color: #f9fafb; color: #111827; }
        .stTextInput>div>div>input { background-color: #ffffff !important; color: #111827 !important; border-radius: 12px !important; border: 1px solid #d1d5db !important; }
        """
    elif mode == "Solid":
        css += f"""
        .stApp {{ background-color: {solid_color}; color: white; }}
        .stTextInput>div>div>input {{ background-color: rgba(255,255,255,0.15) !important; color: white !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.3) !important; }}
        section[data-testid="stSidebar"] {{ background-color: rgba(0,0,0,0.3); }}
        """
    elif mode == "Gradient":
        css += """
        .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); color: white; }
        .stTextInput>div>div>input { background: rgba(255,255,255,0.15) !important; backdrop-filter: blur(10px); color: white !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.3) !important; }
        """
    
    css += "</style>"
    st.markdown(css, unsafe_allow_html=True)

# 17 COLORS
SOLID_COLORS = {
    "🩶 Grey": "#9ca3af",
    "❤️ Red": "#dc2626", 
    "🧡 Orange": "#f97316",
    "🟡 Yellow": "#eab308",
    "🟢 Green": "#16a34a",
    "🔵 Blue": "#2563eb",
    "🟣 Purple": "#7c3aed",
    "🟤 Brown": "#92400e",
    "🌸 Pink": "#ec4899",
    "⚪ Cream": "#fef3c7",
    "💚 Mint": "#4ade80",
    "💙 Sky": "#60a5fa",
    "💜 Lavender": "#c084fc",
    "⚫ Black": "#111827",
    "⚪ White": "#ffffff",
    "🟢 Lime": "#84cc16",
    "🔵 Navy": "#1e40af"
}

# SIDEBAR THEME CONTROLS
st.sidebar.markdown("## 🎨 **Theme**")
mode = st.sidebar.selectbox("Mode", ["Dark", "Light", "Solid", "Gradient"], index=0)

if mode == "Solid":
    st.sidebar.markdown("### 🟡 **Solid Colors**")
    selected_color_name = st.sidebar.selectbox("Pick color", list(SOLID_COLORS.keys()), format_func=lambda x: x)
    solid_color = SOLID_COLORS[selected_color_name]
else:
    solid_color = "#6366f1"

# Apply theme
set_theme(mode, solid_color)

# MAIN APP
st.set_page_config(page_title="CBSE Science Bot", page_icon="📘", layout="wide")

st.markdown("# 📘 **CBSE Class 10 Science Bot**")
st.markdown("**Biology • Chemistry • Physics**")

# Subject cards (ALWAYS black text now)
col1, col2, col3 = st.columns(3)
with col1: 
    st.info("**🧬 Life Processes**")
with col2: 
    st.info("**🧪 Acids & Salts**")
with col3: 
    st.info("**🔬 Light Reflection**")

st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("💬 Ask a science question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        best_intent, confidence = find_best_intent(prompt)
        if confidence > 0.3 and best_intent:
            response = f"**✅ {best_intent['subject']}** • {best_intent['chapter']}\n\n{best_intent['response']}\n\n*Confidence: {confidence:.2f}*"
            st.success(response)
        else:
            st.error("❌ Try: 'photosynthesis', 'baking soda uses'")
        st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()
