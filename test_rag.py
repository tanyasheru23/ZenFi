from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from answer_generator import generate_answer

MAX_HISTORY_MESSAGES = 4   # last 4 turns
SUMMARY_TRIGGER = 6       # when to summarize

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectordb = FAISS.load_local(
    "vectordb",
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# MEMORY STORAGE
# -----------------------------
full_history = []
memory_summary = ""


# -----------------------------
# MEMORY SUMMARIZER
# -----------------------------
def summarize_history(history):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    history_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in history]
    )

    prompt = f"""
Summarize this conversation into short useful memory
for a financial assistant.

Keep only:
- salary or income info
- financial goals
- investment stage
- important preferences

Do NOT include detailed explanations.

Conversation:
{history_text}
"""

    response = llm.invoke(prompt)
    return response.content


# -----------------------------
# CHAT LOOP
# -----------------------------
while True:
    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    # only send recent history
    recent_history = full_history[-MAX_HISTORY_MESSAGES:]

    # Generate answer
    answer = generate_answer(
        vectordb,
        query,
        recent_history,
        memory_summary
    )

    print("\nAnswer:\n", answer)

    # Update history
    full_history.append({"role": "user", "content": query})
    full_history.append({"role": "assistant", "content": answer})

    # ---- SUMMARIZE IF TOO LONG ----
    if len(full_history) >= SUMMARY_TRIGGER:

        new_summary = summarize_history(full_history[:-4])

        memory_summary = f"""
        Previous memory:
        {memory_summary}

        Updated memory:
        {new_summary}
        """

        # keep only recent messages
        full_history = full_history[-4:]
