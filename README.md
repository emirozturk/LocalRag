# LocalRag
#### Sample RAG Application for "Practical Applications of Emerging Tech in Data Science" event
(https://www.meetup.com/pydata-turkiye/events/302866344/)

query_data.py: Sets up a system that searches a vector database (Chroma) for relevant documents using embeddings from the Ollama model, and generates a context-based answer to user queries with the "Llama 3.1" model. It retrieves the most similar documents, constructs a context, and invokes a language model to answer the query based on that context.

update_database.py: Loads PDFs, splits their content into chunks, and saves the chunks in a `Chroma` vector database using `OllamaEmbeddings` for efficient document retrieval. It processes the PDFs and stores the embedded text for future similarity searches.
