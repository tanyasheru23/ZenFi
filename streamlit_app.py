import streamlit as st

from answer_generator import generate_answer
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


st.set_page_config(page_title="ZenFi", layout="wide")
st.title("ZenFi — Financial Learning Assistant")


# -------------------------
# LOAD VECTOR DB (RUN ONCE)
# -------------------------
@st.cache_resource
def load_vectordb():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectordb = FAISS.load_local(
        "vectordb",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectordb


vectordb = load_vectordb()


# -------------------------
# SESSION MEMORY
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory_summary" not in st.session_state:
    st.session_state.memory_summary = ""


# -------------------------
# DISPLAY OLD MESSAGES FIRST
# -------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------
# USER INPUT
# -------------------------
query = st.chat_input("Ask about budgeting, savings, taxes...")

if query:

    # show user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    recent_history = st.session_state.chat_history[-4:]

    response = generate_answer(
        vectordb,
        query,
        recent_history,
        st.session_state.memory_summary
    )

    # save messages
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })

    # show assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
