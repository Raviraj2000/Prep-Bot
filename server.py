from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from questions.questions import get_interview_question
# from transcription.whisper_transcribe import get_transcription
from database.database import get_relevant_text
# from groq_api import evaluate
from typing import Optional

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
async def get_question():
    try:
        question = get_interview_question()
        return {"question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
async def evaluate_response(question: str = Form(...),candidate_answer: str = Form(...)):
    try:
        if not question or not candidate_answer:
            raise HTTPException(status_code=400, detail="Missing question or candidate answer.")
        
        print(f"Question: {question}")
        print(f"Candidate Answer: {candidate_answer}")
        
        relevant_data = get_relevant_text(question)
        # response = evaluate(question, relevant_data, candidate_answer)
        print(relevant_data)
        
        return relevant_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
