from langchain.text_splitter import RecursiveCharacterTextSplitter
import re, json
import tiktoken

# Load Text
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

# Clean Text to reduce noise and size of data
def clean_text(text):
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Chunk Text to reduce the size of data
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

#Add metadata for better traceability
def add_metadata(chunks):
    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        chunks_with_metadata.append({
            "text": chunk,
            "metadata": {
                "source": "301-smart-answers",
                "chunk_id": f"chunk_{i+1}",
                "position": i,
                "length": len(chunk.split())  # Optional: Word count of the chunk
            }
        })
    print(f"✅ Added unique metadata to {len(chunks_with_metadata)} chunks.")
    return chunks_with_metadata

# Validate Tokens to ensure that the chunks are within the token limit
def validate_chunks(chunks):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return [chunk for chunk in chunks if len(tokenizer.encode(chunk["text"])) <= 8192]

# Save Chunks to a file
def save_chunks(chunks, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')

# Main Pipeline 
def main():
    text = load_text('data/301 smart answers to tough interview questions - Vicky Oliver.txt')
    text = clean_text(text)
    chunks = chunk_text(text)
    chunks_with_metadata = add_metadata(chunks)
    valid_chunks = validate_chunks(chunks_with_metadata)
    save_chunks(valid_chunks, 'data/cleaned_interview_book.jsonl')

if __name__ == '__main__':
    main()
