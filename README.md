# ZenFi — Conversational AI Financial Learning Assistant

ZenFi is a conversational AI assistant designed to help beginners understand personal finance concepts in a structured, calm, and practical way. The system uses Retrieval-Augmented Generation (RAG) to provide grounded responses based on curated financial education resources instead of relying solely on model knowledge.

The goal of ZenFi is to make financial learning less overwhelming for early-career individuals by providing step-by-step explanations, budgeting templates, and structured financial guidance.

---

## 🚀 Features

- Retrieval-Augmented Generation (RAG) pipeline for grounded responses
- Conversational memory for multi-turn interactions
- Context compression to reduce redundant retrieval
- Beginner-friendly financial explanations
- Structured outputs with steps and tables
- Safety-focused responses prioritizing emergency funds and debt management
- Streamlit-based interactive chat interface
- FastAPI backend for API-based integration

---

## 🧠 System Overview

ZenFi follows a modular AI system design:

```
User Query
↓
Retriever (FAISS Vector Database)
↓
Context Compression
↓
Conversation Memory (Summary + Recent History)
↓
LLM Response Generation
↓
Streamlit Chat Interface
```


The assistant retrieves relevant financial information from curated documents and generates responses using only retrieved context.

---

## 🏗️ Architecture

### Components

#### Document Processing
- Financial education PDFs from RBI, SEBI, Income Tax Department
- Text extraction and chunking
- Embedding generation

#### Vector Database
- FAISS for semantic similarity search
- OpenAI embeddings

#### Retrieval Layer
- Top-k semantic retrieval
- Duplicate context filtering

#### Reasoning Layer
- Prompt-controlled response generation
- Conversational memory with summarization

#### Interface
- Streamlit chat UI
- FastAPI backend endpoints

---

## 🧰 Tech Stack

- Python
- LangChain
- OpenAI API
- FAISS (Vector Database)
- FastAPI
- Streamlit
- PyMuPDF (PDF processing)

---

## 📊 Dataset

ZenFi uses publicly available financial literacy resources including:

- Financial education handbooks
- Mutual fund guides
- Taxation guides for salaried individuals
- Personal finance primers

These documents are used only for educational retrieval purposes.

---

## ▶️ Running the Project

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/zenfi.git
cd zenfi
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Set Environment Variables

Create a .env file:

```bash
OPENAI_API_KEY=your_api_key
```

4. Run Streamlit App
```bahs
streamlit run streamlit_app.py
```

🔌 FastAPI Backend (Optional)

Run API server:

```bash
uvicorn main:app --reload
```


## ✅ Future Improvements

Tool integration for financial calculators

Multimodal ingestion (slides, audio transcripts)

User session-based memory

Agent-based planning workflows

Deployment using Docker containers
