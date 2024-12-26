import logging
from pymilvus import MilvusClient
import redis
import random

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ✅ Initialize Redis Client
redis_client = redis.StrictRedis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

# ✅ Initialize Milvus Client
milvus_client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

TO_ASK_KEY = "questions:to_ask"
ASKED_KEY = "questions:asked"
QUESTION_TTL = 1800 


def initialize_questions():
    logging.info("Initializing questions...")
    try:
        milvus_client.load_collection(collection_name="interview_questions")
        results = milvus_client.query(
            collection_name="interview_questions",
            filter="id >= 0", 
            output_fields=["text"],
            limit=1000
        )
        
        logging.debug(f"Fetched {len(results)} questions from Milvus.")
    
        if not redis_client.exists(TO_ASK_KEY):
            for question in results:
                redis_client.sadd(TO_ASK_KEY, question['text'])
            logging.info("All questions loaded from Milvus into Redis.")
        else:
            logging.info("Redis cache already populated with questions.")
    except Exception as e:
        logging.error(f"Failed to load questions from Milvus: {e}")

def clean_expired_questions():
    """
    Remove expired questions from the ASKED_KEY set based on TTL.
    """
    try:
        expired_questions = []
        asked_questions = redis_client.smembers(ASKED_KEY)

        for question in asked_questions:
            ttl = redis_client.ttl(f"{ASKED_KEY}:{question}")
            if ttl == -2:  # Key has expired
                redis_client.srem(ASKED_KEY, question)
                redis_client.sadd(TO_ASK_KEY, question)
    except Exception as e:
        logging.error(f"Failed to clean expired questions: {e}")

def get_interview_question():
    """
    Fetch a random question from Redis without repetition.
    Automatically resets if all questions have been asked.
    """
    try:
        # Clean expired questions before fetching
        clean_expired_questions()

        # Check if `to_ask` is empty
        if redis_client.scard(TO_ASK_KEY) == 0:
            # Automatically reset from `asked` to `to_ask`
            asked_questions = redis_client.smembers(ASKED_KEY)
            if asked_questions:
                for question in asked_questions:
                    redis_client.sadd(TO_ASK_KEY, question)
                redis_client.delete(ASKED_KEY)
                logging.info("♻️ All asked questions moved back to 'to_ask'.")
            else:
                logging.info("⚠️ No questions available in both 'to_ask' and 'asked'.")
                return None

        # Fetch a random question from `to_ask`
        question = redis_client.srandmember(TO_ASK_KEY)
        if not question:
            return None

        # Move to `asked` with TTL
        redis_client.srem(TO_ASK_KEY, question)
        redis_client.sadd(ASKED_KEY, question)
        redis_client.setex(f"{ASKED_KEY}:{question}", QUESTION_TTL, question)

        return question
    
    except Exception as e:
        logging.error(f"❌ Failed to fetch a question: {e}")
        return None

# ✅ Reset Questions
def reset_questions():
    try:
        redis_client.delete(TO_ASK_KEY)
        redis_client.delete(ASKED_KEY)
        initialize_questions()
        logging.info("Questions reset successfully.")
    except Exception as e:
        logging.error(f"Failed to reset questions: {e}")

initialize_questions()