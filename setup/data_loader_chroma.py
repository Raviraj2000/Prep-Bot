from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
import os
from langchain_openai import OpenAIEmbeddings
import dotenv
dotenv.load_dotenv()

txt_path = 'data/301 smart answers to tough interview questions - Vicky Oliver.txt'

loader = UnstructuredFileLoader(txt_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunked_documents = text_splitter.split_documents(docs)

# Create and persist the Chroma vector database
vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(),
        persist_directory="data/chroma_store" 
)

vectordb.persist()  # Persist the vector database