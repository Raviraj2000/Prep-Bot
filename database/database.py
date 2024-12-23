import json
from pymilvus import MilvusClient
from transformers import AutoTokenizer, AutoModel
import torch

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
            print(f"❌ Failed to generate embedding: {e}")
            return None

class MilvusDatabase:
    def __init__(self, uri="http://127.0.0.1:19530", token="root:Milvus"):
        try:
            self.client = MilvusClient(uri=uri, token=token)
            print("✅ Successfully connected to Milvus.")
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {e}")
            self.client = None

    def query(self, query_embedding, collection_name="interview_book_bge", limit=5):
        if not self.client:
            print("❌ Milvus client is not initialized.")
            return []
        
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 128}
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
            print(f"✅ Retrieved {len(retrieved_texts)} relevant chunks from Milvus.")
            return retrieved_texts
        
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []

model = BGEEmbedModel()
milvus_db = MilvusDatabase()

def get_relevant_text(question):
    if model is None or milvus_db is None:
        print("❌ Model or Milvus database instance is missing.")
        return []

    # Generate query embedding
    query_embedding = model.get_embedding(question)
    if not query_embedding:
        print("❌ Failed to generate embedding for the question.")
        return []

    # Query Milvus
    relevant_texts = milvus_db.query(query_embedding)
    return relevant_texts[0]