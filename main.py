from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


from answer_generator import generate_answer

app = FastAPI()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectordb = FAISS.load_local(
    "vectordb",
    embeddings,
    allow_dangerous_deserialization=True
)

chat_history = []
memory_summary = ""



class Query(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "ZenFi API running"}


@app.post("/chat")
def chat(q: Query):

    global chat_history
    global memory_summary

    recent_history = chat_history[-4:]

    response = generate_answer(
        vectordb,
        q.question,
        recent_history,
        memory_summary
    )

    # update history
    chat_history.append({
        "role": "user",
        "content": q.question
    })

    chat_history.append({
        "role": "assistant",
        "content": response
    })

    return {"answer": response}

