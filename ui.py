import streamlit as st
from app import get_answer

st.title("PDF RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (always at bottom)
user_query = st.chat_input("Ask a question from the PDF")

if user_query:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    # Call your RAG function here
    answer = get_answer(user_query)

    # Show assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
