import json
from transformers import AutoTokenizer, AutoModel
import torch

# ✅ Step 1: Load Cleaned Chunks
def load_chunks(file_path):
    """
    Load cleaned text chunks from a JSONL file.
    """
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"✅ Loaded {len(chunks)} chunks from {file_path}.")
    return chunks


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



# ✅ Step 3: Generate and Save Embeddings
def generate_embeddings(chunks, embedding_model, output_file):
    """
    Generate embeddings for all text chunks and save them with metadata.
    """
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.get_embedding(chunk["text"])
        if embedding:
            embedded_chunks.append({
                "id": i,
                "text": chunk["text"],
                "embedding": embedding,
                "metadata": chunk["metadata"]
            })
        else:
            print(f"❌ Skipped chunk {i} due to embedding failure.")
    
    # Save embeddings to a JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in embedded_chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f"✅ Generated embeddings for {len(embedded_chunks)} chunks and saved to {output_file}.")


# ✅ Step 4: Main Function
def main():
    input_file = 'data/cleaned_interview_book.jsonl'
    output_file = 'data/embedded_interview_book_bge.jsonl'
    
    # Load chunks
    chunks = load_chunks(input_file)
    
    # Initialize BGE Model
    embedding_model = BGEEmbedModel()
    
    # Generate and Save Embeddings
    generate_embeddings(chunks, embedding_model, output_file)


# ✅ Execute Script
if __name__ == '__main__':
    main()
