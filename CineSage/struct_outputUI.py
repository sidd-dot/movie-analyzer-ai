import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
import json

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
    Release_Year: Optional[int]
    Genre: List[str]
    Rating: Optional[str]
    Director: str
    Cast: List[str]
    Plot: str
    Key_Themes: List[str]
    Summary: str

parser = PydanticOutputParser(pydantic_object=MovieInfo)

# ---------- CLEAN FUNCTION ----------
def clean_json(text):
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            if "{" in p:
                text = p
                break
    text = text[text.find("{"): text.rfind("}") + 1]
    return text.strip()

# ---------- PROMPT ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert movie information extractor.

STRICT RULES:
- ONLY return valid JSON
- No markdown
- Missing → "NA"

{format_instructions}
"""),
    ("human", "Extract from:\n\n{paragraph}")
])

# ---------- CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body {font-family: 'Poppins', sans-serif;}

.stApp {
    background-image: url("https://wallpapercave.com/wp/wp1945897.jpg");
    background-size: cover;
}

.block-container {
    max-width: 920px;
    margin: auto;
    margin-top: 17vh;
    padding: 35px;
    background: rgba(25,25,25,0.25);
    border-radius: 18px;
    backdrop-filter: blur(14px);
}

.title {
    font-size: 2.8rem;
    text-align: center;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    color: #ccc;
    margin-bottom: 25px;
}

.json-box {
    margin-top: 15px;
    padding: 10px;
    border-radius: 10px;
    background: rgba(0,0,0,0.6);
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
user_input = st.text_area("📝 Enter Movie Description", height=200, value=st.session_state.text)
st.session_state.text = user_input

col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🚀 Analyze")

with col2:
    clear = st.button("🧹 Clear")

# ---------- CLEAR ----------
if clear:
    st.session_state.clear()
    st.rerun()

# ---------- PROCESS ----------
if analyze and user_input.strip():

    with st.spinner("🎬 Generating structured JSON..."):

        final_prompt = prompt.invoke({
            "paragraph": user_input,
            "format_instructions": parser.get_format_instructions()
        })

        response = model.invoke(final_prompt)
        cleaned = clean_json(response.content)

        output_data = None
        status = ""

        # ---------- PYDANTIC ----------
        try:
            data = parser.parse(cleaned)
            output_data = data.model_dump()
            status = "✅ Schema validated"

        # ---------- FALLBACK ----------
        except:
            try:
                output_data = json.loads(cleaned)
                status = "⚠️ Schema mismatch — recovered JSON"
            except:
                st.error("❌ Could not parse output")
                st.stop()

        # ---------- DISPLAY ----------
        st.markdown("### 📦 Structured JSON Output")
        st.info(status)

        st.markdown('<div class="json-box">', unsafe_allow_html=True)
        st.json(output_data)
        st.markdown('</div>', unsafe_allow_html=True)

        st.download_button(
            "📥 Download JSON",
            json.dumps(output_data, indent=2),
            file_name="movie_data.json"
        )