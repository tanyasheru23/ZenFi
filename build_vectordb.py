import os
import fitz
import pandas as pd

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from utils import is_template_chunk, table_to_text

load_dotenv()

PDF_FOLDER = "data/pdfs"
VECTOR_DB_PATH = "vectordb"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


# -----------------------------
# TABLE EXTRACTION
# -----------------------------
def extract_tables_from_pdf(pdf_path):

    table_docs = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        tables = page.find_tables()

        if tables.tables:
            for table in tables.tables:
                df = table.to_pandas()
                df = df.replace("\n", " ", regex=True)

                text_version = table_to_text(df)

                table_docs.append(
                    Document(
                        page_content=text_version,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "type": "table",
                            "page": page_num
                        }
                    )
                )

    return table_docs


# -----------------------------
# TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    text_docs = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()

        if len(text.strip()) < 50:
            continue

        text_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "type": "text",
                    "page": page_num
                }
            )
        )

    return text_docs


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def load_and_filter_documents():

    all_documents = []

    for file in os.listdir(PDF_FOLDER):

        if not file.endswith(".pdf"):
            continue

        file_path = os.path.join(PDF_FOLDER, file)
        print(f"Processing {file}")

        # Extract text + tables
        text_docs = extract_text_from_pdf(file_path)
        table_docs = extract_tables_from_pdf(file_path)

        documents = text_docs + table_docs

        # Chunking
        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:

            text = chunk.page_content

            # Keep only actionable/template content
            if not is_template_chunk(text):
                continue

            chunk.metadata["category"] = "financial_template"
            all_documents.append(chunk)

    print(f"Total filtered chunks: {len(all_documents)}")
    return all_documents


def build_vectordb():

    docs = load_and_filter_documents()

    print("Creating FAISS index...")
    vectordb = FAISS.from_documents(docs, embeddings)

    vectordb.save_local(VECTOR_DB_PATH)

    print("VectorDB saved successfully!")


if __name__ == "__main__":
    build_vectordb()
