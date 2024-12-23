from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from questions.questions import get_interview_question
from database.database import get_relevant_data
from groq_api import evaluate
from typing import Optional
import uvicorn
import logging
from logging.handlers import RotatingFileHandler
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("logs/fastapi_app.log", maxBytes=1000000, backupCount=3),
        logging.StreamHandler()
    ]
)

# Create Logger
logger = logging.getLogger("fastapi_app")

app = FastAPI(title="Interview Prep API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"message": "Welcome to Interview Prep"}

@app.get("/api/question")
def get_question():
    try:
        question = get_interview_question()
        return question
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
def evaluate_response(question: str = Form(...),candidate_answer: str = Form(...)):
    try:
        if not question:
            raise HTTPException(status_code=400, detail="Missing question")
        if not candidate_answer:
            raise HTTPException(status_code=400, detail="Missing answer")
        
        relevant_data = get_relevant_data(question)
        response = evaluate(question, relevant_data, candidate_answer)
        response = json.loads(response)

        logger.info(f"Question: {question}")
        logger.info(f"Answer: {candidate_answer}")
        logger.info(f"Feedback: {response}")
        
        if not response:
            response = {
                "Feedback": {
                    "Strengths": ["No strengths identified"],
                    "Areas for Improvement": ["No areas for improvement identified"],
                    "Suggestions for Improvement": ["No suggestions for improvement identified"]
                }
            }
        if "Feedback" not in response:
            logger.Info("Feedback not in response")
            response["Feedback"] = {
                "Strengths": ["No strengths identified"],
                "Areas for Improvement": ["No areas for improvement identified"],
                "Suggestions for Improvement": ["No suggestions for improvement identified"]
            }
        else:
            logger.info("Feedback in response")
            if "Strengths" not in response["Feedback"] or not response["Feedback"]["Strengths"]:
                logger.info("Strengths not in response")
                response["Feedback"]["Strengths"] = ["No strengths identified"]
            if "Areas for Improvement" not in response["Feedback"] or not response["Feedback"]["Areas for Improvement"]:
                logger.info("AFI not in response")
                response["Feedback"]["Areas for Improvement"] = ["No areas for improvement identified"]
            if "Suggestions for Improvement" not in response["Feedback"] or not response["Feedback"]["Suggestions for Improvement"]:
                logger.info("SFI not in response")
                response["Feedback"]["Suggestions for Improvement"] = ["No suggestions for improvement identified"]

        logger.info("Response validated!")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str("I am being hit"))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)