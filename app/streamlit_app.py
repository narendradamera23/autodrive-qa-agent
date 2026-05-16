import streamlit as st
import requests

import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoDrive QA Agent",
    page_icon="🚗",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────────────
st.title("AutoDrive QA Agent")
st.markdown("Ask anything about your vehicle — fault codes, maintenance, features, and more.")
st.divider()

# ── Session State ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    st.markdown("**Model:** llama3.2 (local)")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**Knowledge Base:** Toyota Camry 2022")
    st.divider()

    if st.button("Clear Chat History", use_container_width=True):
        try:
            requests.delete(f"{API_URL}/reset")
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()
        except:
            st.error("Could not connect to API")

    st.divider()
    st.markdown("**Example Questions:**")
    st.markdown("- What does P0420 mean?")
    st.markdown("- How do I check engine oil?")
    st.markdown("- What is AWD vs FWD?")
    st.markdown("- Explain fault code P0300")
    st.markdown("- How does the AC system work?")

# ── Chat History Display ──────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your vehicle..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt},
                    timeout=120
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                else:
                    st.error(f"API error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI is running.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The model is taking too long.")