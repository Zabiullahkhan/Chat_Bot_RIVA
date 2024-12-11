from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Step 1: Text Chunking - Split the data into small chunks
def text_chunking(data, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(data)
    return chunks

# Step 2: Embedding Transformation - Use an embedding model to convert chunks into embeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Step 3: Vector Storage - Initialize vector database connection
pgvector = PGVector(
    connection_string="postgresql://user:password@localhost:5432/vector_db",
    table_name="documents",
    embedding_function=embedding_model.embed_query
)

# Function to populate the database
def store_embeddings(chunks):
    for chunk in chunks:
        pgvector.add_texts([chunk])

# Step 4: Text Mapping - Save the text chunks for future reference
# Done implicitly by storing them in the PGVector database

# Step 5: Query Embedding - Transform the user's query into an embedding
# Handled implicitly by LangChain's retrieval system

# Step 6: Database Querying & Step 7: Similarity Search
retriever = pgvector.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Step 8: Context Mapping - Map the retrieved vectors to text chunks
# Done implicitly by the retriever

# Step 9: LLM Response - Use the retrieved context in the prompt
chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.0)

prompt_template = """
You are a helpful assistant. Use the provided context to answer the user's question.

Context: {context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=retriever,
    return_source_documents=True
)

# Example Data and Query
if __name__ == "__main__":
    # Example data
    data = """
    LangChain provides a framework to build applications powered by language models.
    PG Vector is a vector database plugin for PostgreSQL, enabling similarity search.
    """

    # Step 1: Chunk the text
    chunks = text_chunking(data)

    # Step 2-4: Transform and store embeddings in the database
    store_embeddings(chunks)

    # User query
    query = "What is PG Vector?"

    # Step 9: Get the response from the LLM
    response = qa_chain.run(query)

    print("Response:", response)
