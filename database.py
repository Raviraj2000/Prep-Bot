from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vector_db = Chroma(
        persist_directory='data/chroma_store/',
        embedding_function=OpenAIEmbeddings()
    )

retriever = vector_db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6}
)

def get_relevant_data(question):
    
    similar_docs = retriever.get_relevant_documents(question)
    retrieved_data = similar_docs[0].page_content if similar_docs else "No relevant data found."

    return retrieved_data