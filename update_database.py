from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma.vectorstores import Chroma
import os


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def save_to_chroma(db_path,chunks: list[Document]):
    if os.path.exists(db_path):
        os.mkdir(db_path)
        db = Chroma(persist_directory=db_path, embedding_function=OllamaEmbeddings(model="llama3.1"),collection_metadata={"hnsw:space": "cosine"})
        print(f"Loaded existing database from {db_path}.")
        db.add_documents(chunks)
    else:
        db = Chroma.from_documents(chunks, OllamaEmbeddings(model="llama3.1"), persist_directory=db_path)
        print(f"Created new database and saved {len(chunks)} chunks to {db_path}.")
    print(f"Added {len(chunks)} chunks to the database at {db_path}.")
    

def get_documents(path):
    files=os.listdir(path)
    list_of_documents = []
    for file in files:
        if "pdf" in file:
            loader = PyPDFLoader(os.path.join(path,file))
            documents = loader.load()
            list_of_documents.extend(documents)
    return list_of_documents


db_path = "/Users/emirozturk/Desktop/GitHub/LocalRag/Chroma"
docs_path = "/Users/emirozturk/Desktop/Pdf/"


documents = get_documents(docs_path)
chunks = split_text(documents)
save_to_chroma(db_path,chunks)