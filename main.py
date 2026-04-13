import streamlit as st
from src.chains.rag_chain import chain, chat_history
from langchain.messages import HumanMessage, AIMessage
import time

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("💬 RAG Chatbot")

# ---------------------------
# Session State Init
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything 👋"}
    ]

# ---------------------------
# Display Chat History
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# User Input
# ---------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Add to backend history
    chat_history.append(HumanMessage(user_input))

    # ---------------------------
    # Streaming Response (Fake Typing Effect)
    # ---------------------------
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            response = chain.invoke(user_input)

        # Typing effect (word by word)
        for word in response.split():
            full_response += word + " "
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.03)  # speed control

        message_placeholder.markdown(full_response)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    chat_history.append(AIMessage(full_response))


# ---------------------------
# Sidebar (Better UX)
# ---------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. Ask again!"}
        ]
        chat_history.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("- Ask specific questions\n- Avoid vague queries\n- Use context if needed")