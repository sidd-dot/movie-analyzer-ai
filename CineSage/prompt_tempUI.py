import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# ---------- LOAD ENV ----------
load_dotenv()

st.set_page_config(page_title="🎬 CineSage AI", layout="wide")

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return ChatMistralAI(model="mistral-small-2603")

model = load_model()

# ---------- PROMPT ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert movie information extractor and summarizer.

STRICT RULES:
- No hallucination
- Only extract given info
- Missing → Not Available

Format exactly:

Movie Name:
Genre:
Release Year:
Director:
Cast:
Plot:
Key Themes:
Rating:
Notable Features:
Short Summary:
"""),
    ("human", "Extract from:\n\n{paragraph}")
])

# ---------- CSS ----------
st.markdown("""
<style>

/* ---------- FONT ---------- */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* ---------- BACKGROUND ---------- */
.stApp {
    background-image: url("https://wallpapercave.com/wp/wp1945897.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ---------- MAIN CONTAINER ---------- */
.block-container {
    max-width: 900px;
    margin: auto;
    margin-top: 12vh;
    margin-bottom: 10vh;
    padding: 40px;

    background: rgba(20,20,20,0.35);
    border-radius: 20px;

    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);

    box-shadow: 
        0 15px 60px rgba(0,0,0,0.8),
        0 0 30px rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);
}

/* ---------- TITLE ---------- */
.title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    color: white;
    text-shadow: 0 0 25px rgba(255,255,255,0.25);
}

/* ---------- SUBTITLE ---------- */
.subtitle {
    text-align: center;
    color: #cccccc;
    margin-bottom: 30px;
    font-weight: 300;
}

/* ---------- LABEL ---------- */
.label {
    margin-bottom: 10px;
    color: #eeeeee;
}

/* ---------- TEXT AREA (GREY GLASS) ---------- */
textarea {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    padding: 12px !important;
    font-size: 14px !important;
}

/* ---------- BUTTON BASE (BLACK) ---------- */
.stButton>button {
    background: rgba(0,0,0,0.85);
    color: white;
    border-radius: 25px;
    padding: 10px 24px;
    border: 1px solid rgba(255,255,255,0.15);
    font-weight: 600;

    transition: all 0.25s ease;
}

/* ---------- BUTTON HOVER (RED EFFECT) ---------- */
.stButton>button:hover {
    background: linear-gradient(135deg, #ff3c3c, #ff0000);
    color: white;

    transform: translateY(-2px) scale(1.05);

    box-shadow:
        0 0 20px rgba(255,0,0,0.7),
        0 0 40px rgba(255,0,0,0.4);
}

/* ---------- OUTPUT BOX ---------- */
.output-box {
    background: rgba(0,0,0,0.75);
    padding: 20px;
    border-radius: 12px;
    color: #00ffd5;
    font-family: monospace;

    white-space: pre-wrap;   /* 🔥 THIS FIX */
    
    border: 1px solid rgba(255,255,255,0.1);
    margin-top: 15px;
}
}

/* ---------- SECTION TITLE ---------- */
.section-title {
    font-size: 1.5rem;
    margin-top: 25px;
    font-weight: 600;
    color: white;
}

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.2);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------- SESSION ----------
if "text" not in st.session_state:
    st.session_state.text = ""

# ---------- HEADER ----------
st.markdown('<div class="title">🎬 CineSage AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Movie Intelligence Engine</div>', unsafe_allow_html=True)

# ---------- INPUT ----------
st.markdown('<div class="label">📝 Enter Movie Description</div>', unsafe_allow_html=True)

user_input = st.text_area(
    "",
    height=200,
    value=st.session_state.text
)

st.session_state.text = user_input

col1, col2 = st.columns(2)

with col1:
    extract = st.button("🚀 Analyze Movie")

with col2:
    clear = st.button("🧹 Clear")

# ---------- CLEAR ----------
if clear:
    st.session_state.text = ""
    st.rerun()

# ---------- PROCESS ----------
if extract:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("🎬 Extracting cinematic insights..."):
            final_prompt = prompt.invoke({"paragraph": user_input})
            response = model.invoke(final_prompt)

        st.markdown('<div class="section-title">🎯 Extracted Information</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="output-box">{response.content}</div>',
            unsafe_allow_html=True
        )

        st.download_button(
            "📥 Download Result",
            response.content,
            file_name="movie_info.txt"
        )