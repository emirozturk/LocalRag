# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil


CHROMA_PATH = "chroma"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader("/Users/emirozturk/Desktop/Data", glob="*.txt")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Check if the database already exists
    if os.path.exists(CHROMA_PATH):
        # Load the existing database
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(model="llama3"))
        print(f"Loaded existing database from {CHROMA_PATH}.")
    else:
        # Create a new database
        db = Chroma.from_documents(
            chunks, OllamaEmbeddings(model="llama3"), persist_directory=CHROMA_PATH
        )
        print(f"Created new database and saved {len(chunks)} chunks to {CHROMA_PATH}.")
    
    # Add new documents to the existing or new database
    db.add_documents(chunks)
    db.persist()
    print(f"Added {len(chunks)} chunks to the database at {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
