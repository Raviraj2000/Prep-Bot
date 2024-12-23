from pymilvus import MilvusClient
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# ✅ Step 1: Connect to Milvus
client = MilvusClient(
    uri="http://127.0.0.1:19530",
    token="root:Milvus"
)
print("Successfully connected to Milvus.")

import torch
from transformers import AutoTokenizer, AutoModel

class BGEEmbedModel:
    def __init__(self, model_name="BAAI/bge-large-en"):
        """
        Initialize BGE Embedding model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_embedding(self, text):
        """
        Generate embeddings for a given text using mean pooling.
        """
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use mean pooling across tokens
            embedding = outputs.last_hidden_state.mean(dim=1)
            # Normalize embedding for COSINE similarity
            normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return normalized_embedding[0].numpy().tolist()
        except Exception as e:
            print(f"❌ Failed to generate embedding: {e}")
            return None


# ✅ Step 3: Query Milvus for Similar Chunks
def query_milvus(client, query_embedding):
    """
    Query Milvus for the most similar chunk based on embedding.
    """
    try:
        # Perform search with correct parameter structure
        results = client.search(
            collection_name="interview_book_bge",
            data=[query_embedding],
            anns_field="embedding",
            search_params={
                "metric_type": "COSINE",
                "params": {
                    "nprobe": 64  # Search across 64 clusters for balanced accuracy and speed
                }
            },
            limit=5,
            output_fields=["text"]
        )
        print(len(results))
        print("\n==== QUERY RESULTS ====\n")

        for i in range(len(results)):
            print(f"\nRank {i+1}: ID: {results[i][0]['id']}, Distance: {results[i][0]['distance']}, Text: {results[i][0]['entity']['text']}\n\n")

    except Exception as e:
        print(f"❌ Query failed: {e}")


# ✅ Step 4: Run Query
def main():
    sample_query = "Do you think that the past is any precedent for the future?"
    print(f"\n==== QUERYING MILVUS FOR: '{sample_query}' ====")
    
    # Initialize embedding model
    embedding_model = BGEEmbedModel()
    
    # Generate embedding
    query_embedding = embedding_model.get_embedding(sample_query)
    if query_embedding:
        query_milvus(client, query_embedding)
    else:
        print("❌ Failed to generate embedding for the query.")


if __name__ == '__main__':
    main()
