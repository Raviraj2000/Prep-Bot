import re
import json
import os
import spacy
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_text(file_path):
    """
    Reads the entire file content as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def clean_text(text):
    """
    Removes page numbers, extra newlines, non-ASCII chars, etc.
    """
    # Remove patterns like 'Page X'
    text = re.sub(r'Page \d+', '', text)
    # Collapses multiple newlines to a single newline
    text = re.sub(r'\n+', '\n', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(text):
    """
    Uses spaCy to:
     - convert to lowercase
     - lemmatize words
    This may lose capitalization info (names, acronyms), so use with caution.
    """
    # Load spaCy model for lemmatization
    nlp = spacy.load("en_core_web_sm")
    # Convert to spaCy doc
    doc = nlp(text)
    # Lemmatize and lowercase
    normalized_tokens = [token.lemma_.lower().strip() for token in doc]
    # Rebuild into a single string
    normalized_text = " ".join(normalized_tokens)
    return normalized_text

def chunk_text(text):
    """
    Splits the cleaned+normalized text into smaller chunks.
    Uses LangChain's RecursiveCharacterTextSplitter with chunk_size=1000, overlap=100.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def add_metadata(chunks):
    """
    Converts the list of plain chunk strings into a list of dicts:
    {
      "text": ...,
      "metadata": { ... }
    }
    """
    chunks_with_metadata = []
    for i, chunk in enumerate(chunks):
        chunks_with_metadata.append({
            "text": chunk,
            "metadata": {
                "source": "301-smart-answers",
                "chunk_id": f"chunk_{i+1}",
                "position": i,
                "length": len(chunk.split())  # word count
            }
        })
    return chunks_with_metadata

def remove_boilerplate(chunks, patterns=None):
    """
    Optionally remove chunks containing certain boilerplate (e.g., copyright, disclaimers).
    patterns: list of keywords/phrases you consider "boilerplate."
    If chunk text contains any pattern, exclude it.
    """
    if patterns is None:
        patterns = ["copyright", "all rights reserved", "disclaimer"]
    filtered = []
    for ch in chunks:
        text_lower = ch["text"].lower()
        if not any(p in text_lower for p in patterns):
            filtered.append(ch)
    return filtered

def remove_short_chunks(chunks, min_tokens=50):
    """
    Removes chunks whose token count (by whitespace) is below a given threshold.
    For more precise measurement, we could do a token-based check with tiktoken or spaCy.
    """
    filtered = []
    for ch in chunks:
        word_count = len(ch["text"].split())
        if word_count >= min_tokens:
            filtered.append(ch)
    return filtered

def validate_chunks(chunks):
    """
    Ensures that each chunk does not exceed 8192 tokens (using tiktoken cl100k_base).
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    valid = []
    for ch in chunks:
        token_count = len(tokenizer.encode(ch["text"]))
        if token_count <= 8192:
            valid.append(ch)
    return valid

def save_chunks(chunks, output_file):
    """
    Writes each chunk dict as a JSON line to output_file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')

def main():
    input_file = 'data/301 smart answers to tough interview questions - Vicky Oliver.txt'
    output_file = 'data/cleaned_interview_book.jsonl'

    text = load_text(input_file)
    text = clean_text(text)
    text = normalize_text(text)
    chunks = chunk_text(text)
    chunks_with_metadata = add_metadata(chunks)
    chunks_no_boilerplate = remove_boilerplate(chunks_with_metadata)
    chunks_no_short = remove_short_chunks(chunks_no_boilerplate, min_tokens=50)
    valid_chunks = validate_chunks(chunks_no_short)
    save_chunks(valid_chunks, output_file)

    print(f"✅ Final chunk count: {len(valid_chunks)}")
    print(f"✅ Data preparation complete! Output: {output_file}")

if __name__ == '__main__':
    main()