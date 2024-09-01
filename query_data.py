import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate


PROMPT_TEMPLATE = """Answer the question using only on the given context:
                {context}
                ---
                Answer the question based on the above context: 
                {question}
                """


def main(db_path):
    embedding_function = OllamaEmbeddings(model="llama3")
    db = Chroma(persist_directory=db_path, embedding_function=embedding_function)

    while(True):
        input = input("Query:") 
        if input == ":exit":
            break

        results = db.similarity_search_with_relevance_scores(input, k=3)
        if len(results) == 0:
            print("Sorry, can't find a result")
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=input)

            model = ChatOllama(model="llama3")
            response_text = model.predict(prompt)

            docs = [doc.metadata.get("source", None) for doc, _score in results]
            response = f"Response: {response_text}\nFrom: {docs}"
            print(response)
        
        
db_path = "/Chroma"
main(db_path)