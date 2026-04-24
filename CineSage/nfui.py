import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# ---------- LOAD ENV ----------
load_dotenv()

st.set_page_config(page_title="🎬 CineSage AI", layout="wide")

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return ChatMistralAI(model="mistral-small-2603")

model = load_model()

# ---------- SCHEMA ----------
class MovieInfo(BaseModel):
    Title: str
    Genre: List[str]
    Release_Year: Optional[int]
    Rating: Optional[str]
    Director: str
    Cast: List[str]
    Plot: str
    Key_Themes: List[str]
    Summary: str

parser = PydanticOutputParser(
    pydantic_object=MovieInfo,
    validate_template=False,
    output_format="json"
)

# ---------- PROMPTS ----------
human_prompt = ChatPromptTemplate.from_messages([
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

json_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert movie information extractor.

STRICT RULES:
- ONLY return valid JSON
- No extra text
- Missing → "NA"

{format_instructions}
"""),
    ("human", "Extract from:\n\n{paragraph}")
])

# ---------- CSS (YOUR SAME UI) ----------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background-image: url("https://wallpapercave.com/wp/wp1945897.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container {
    max-width: 850px;
    margin: auto;
    margin-top: 20vh;
    margin-bottom: 10vh;
    padding: 35px;

    background: rgba(25,25,25,0.2);
    border-radius: 18px;

    backdrop-filter: blur(16px);
}

.title {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #dddddd;
    margin-bottom: 25px;
}

.label {
    margin-bottom: 8px;
    color: #eeeeee;
}

textarea {
    background: rgba(0,0,0,0.5) !important;
    color: white !important;
    border-radius: 10px !important;
}

.stButton>button {
    background: linear-gradient(45deg, #ff512f, #dd2476);
    color: white;
    border-radius: 20px;
}

.output-box {
    background: rgba(0,0,0,0.7);
    padding: 15px;
    border-radius: 10px;
    color: #00ffd5;
    font-family: monospace;
    border: 1px solid rgba(255,255,255,0.1);
    margin-top: 15px;
}

.section-title {
    font-size: 1.4rem;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------- SESSION ----------
if "text" not in st.session_state:
    st.session_state.text = ""

# ---------- HEADER ----------
st.markdown('<div class="title">🎬 CineSage AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Movie Intelligence Engine</div>', unsafe_allow_html=True)

# ---------- MODE TOGGLE ----------
mode = st.radio(
    "Select Output Mode",
    ["Human Readable", "Structured JSON"]
)

# ---------- INPUT ----------
st.markdown('<div class="label">📝 Enter Movie Description</div>', unsafe_allow_html=True)

user_input = st.text_area("", height=200, value=st.session_state.text)
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

            if mode == "Human Readable":
                final_prompt = human_prompt.invoke({
                    "paragraph": user_input
                })
                response = model.invoke(final_prompt)
                output = response.content

            else:
                final_prompt = json_prompt.invoke({
                    "paragraph": user_input,
                    "format_instructions": parser.get_format_instructions()
                })

                response = model.invoke(final_prompt)

                try:
                    data = parser.parse(response.content)
                    output = data.json(indent=2)
                except:
                    output = response.content

        st.markdown('<div class="section-title">🎯 Extracted Information</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="output-box">{output}</div>',
            unsafe_allow_html=True
        )

        st.download_button(
            "📥 Download Result",
            output,
            file_name="movie_info.txt"
        )