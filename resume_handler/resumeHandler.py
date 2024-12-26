import re
from PyPDF2 import PdfReader
from typing import Dict
from groq_api.groq_api import parse_resume
import redis
import json

# ✅ Initialize Redis Client
redis_client = redis.StrictRedis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True  # Ensures returned data is in string format
)

# ✅ Store Parsed Resume in Redis
def store_resume_in_redis(user_id: str, parsed_resume: dict):
    try:
        # Convert dictionary to JSON string
        resume_json = json.dumps(parsed_resume)
        # Store in Redis with a unique key
        key = f"resume:{user_id}"
        redis_client.set(key, resume_json)
        print(f"✅ Resume stored in Redis with key: {key}")
    except Exception as e:
        print(f"❌ Failed to store resume in Redis: {e}")

# ✅ Retrieve Parsed Resume from Redis
def get_resume(user_id: str) -> dict:
    try:
        key = f"resume:{user_id}"
        resume_json = redis_client.get(key)
        if resume_json:
            return json.loads(resume_json)
    except Exception as e:
        print(f"❌ Failed to retrieve resume from Redis: {e}")
        return {"No resume exists for the candidate"}

# ✅ Extract text from PDF
def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ✅ Main Function
def handle(pdf_path, name):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    parsed_resume = parse_resume(text)
    store_resume_in_redis(name, parsed_resume)
    return