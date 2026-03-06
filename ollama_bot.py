import streamlit as st
from ollama_engine import OllamaChatEngine

st.set_page_config(page_title="Multi‑Model AI Agent", layout="wide")

# Initialize engine once
if "engine" not in st.session_state:
    st.session_state.engine = OllamaChatEngine()

engine = st.session_state.engine

st.title("🧠 Multi‑Model AI Agent (Ollama + OpenAI + Gemini)")

# ---------------------------------------------------------
# Render full chat history (THIS IS WHERE THE FIX GOES)
# ---------------------------------------------------------
for msg in engine.history:
    role = msg["role"]
    content = msg["content"]

    with st.chat_message(role):

        # Detect base64 images
        if isinstance(content, str) and content.startswith("data:image"):
            st.image(content)

        else:
            st.write(content)

        # If tools were stored inside the message
        if isinstance(msg, dict) and "tools" in msg:
            st.markdown("### Tools used:")
            for t in msg["tools"]:
                st.markdown(f"- `{t}`")

# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.write(user_input)

    # Get agent response
    result = engine.ask(user_input)

    # Show assistant response
    with st.chat_message("assistant"):
        st.write(result["text"])

        # Render image if present
        if result["image"]:
            st.image(result["image"])

        # Render tool usage
        if result["tools"]:
            st.markdown("### Tools used:")
            for t in result["tools"]:
                st.markdown(f"- `{t}`")