import json
from pymilvus import MilvusClient
from transformers import AutoTokenizer, AutoModel
import torch
import os
import certifi
import logging

# Set SSL certificate paths
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BGEEmbedModel:
    def __init__(self, model_name="BAAI/bge-large-en"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_embedding(self, text):
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding[0].numpy().tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

class MilvusDatabase:
    def __init__(self, uri="http://127.0.0.1:19530", token="root:Milvus"):
        try:
            self.client = MilvusClient(uri=uri, token=token)
            logger.info("✅ Successfully connected to Milvus.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.client = None

    def query(self, query_embedding, collection_name="interview_book_bge", limit=5):
        if not self.client:
            logger.error("Milvus client is not initialized.")
            return []
        
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 512}
            }
            
            results = self.client.search(
                collection_name=collection_name,
                data=[query_embedding],
                anns_field="embedding",
                search_params=search_params,
                limit=limit,
                output_fields=["text"]
            )
            
            # Extract text results
            retrieved_texts = [result[0]['entity']['text'] for result in results]
            logger.info(f"Retrieved {len(retrieved_texts)} relevant chunks from Milvus.")
            return retrieved_texts
        
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

def get_relevant_data(question):
    if model is None or milvus_db is None:
        logger.error("Model or Milvus database instance is missing.")
        return []

    # Generate query embedding
    query_embedding = model.get_embedding(question)
    if not query_embedding:
        logger.error("Failed to generate embedding for the question.")
        return []
    
    # Query Milvus
    relevant_texts = milvus_db.query(query_embedding)
    logger.info("Retrieved relevant data from Milvus.")
    return relevant_texts[0]


model = BGEEmbedModel()
milvus_db = MilvusDatabase()