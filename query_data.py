from langchain_chroma.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
import os

PROMPT_TEMPLATE = """Answer the question using only on the given context:
                {context}
                ---
                Answer the question based on the above context: 
                {question}
                """


def main(db_path):
    embedding_function = OllamaEmbeddings(model="llama3.1")
    db = Chroma(persist_directory=db_path, embedding_function=embedding_function)

    while(True):
        inpt = input("Query:") 
        if inpt == ":exit":
            break

        results = db.similarity_search_with_relevance_scores(inpt, k=5)
        if len(results) == 0:
            print("Sorry, can't find a result")
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=inpt)

            model = ChatOllama(model="llama3.1")
            response = model.invoke(prompt)

            docs = [doc.metadata.get("source", None) for doc, _score in results]
            doc_string = "\n".join(docs)
            print(f"Response: {response.content }\n-------------\nFrom:\n{doc_string}\n-------------\n")
        
        
db_path = "/Chroma"
if not os.path.exists(db_path):
    os.makedirs(db_path)
main(db_path)