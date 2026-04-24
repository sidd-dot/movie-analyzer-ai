import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_mistralai import ChatMistralAI
import time

load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Chat Pro",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: auto;
}

.user-msg {
    background-color: #DCF8C6;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: right;
}

.bot-msg {
    background-color: #F1F0F0;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: left;
}

.stChatInput {
    position: fixed;
    bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Control Panel")

    model_choice = st.selectbox(
        "Choose Model",
        ["mistral-small-2603"]
    )

    system_prompt = st.text_area(
        "System Prompt",
        "You are a helpful AI assistant."
    )

    temperature = st.slider("Creativity", 0.0, 1.0, 0.7)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

# ---------------- INIT STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=system_prompt)
    ]

# ---------------- MODEL ----------------
model = ChatMistralAI(
    model=model_choice,
    temperature=temperature
)

# ---------------- HEADER ----------------
st.title("🤖 AI Chat Pro")
st.caption("Interface with memory & streaming")

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---------------- CHAT INPUT ----------------
prompt = st.chat_input("Type your message...")

if prompt:
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response with fake streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        response = model.invoke(st.session_state.messages)

        # Simulate typing effect
        for chunk in response.content.split():
            full_response += chunk + " "
            time.sleep(0.03)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Save response
    st.session_state.messages.append(AIMessage(content=full_response))