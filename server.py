from flask import Flask, jsonify, request
from flask_cors import CORS
from questions.questions import get_interview_question
from transcription.whisper_transcribe import get_transcription
from database import get_relevant_data
from groq_api import evaluate
import json
import base64

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify(message="Welcome to Interview Prep")

@app.route('/api/question')
def get_question():
    question = get_interview_question()
    return jsonify(question)

@app.route('/api/evaluate', methods=['POST'])
def evaluate_response():
    if 'question' not in request.form:
        return jsonify({'error': 'Missing question'}), 400
    
    if 'candidate_answer' not in request.form:
        return jsonify({'error' : 'Missing candidate answer'}), 400

    question = request.form['question']
    print(question)
    candidate_answer = request.form['candidate_answer']
    print(candidate_answer)
    
    relevant_data = get_relevant_data(question)

    response = evaluate(question, relevant_data, candidate_answer)

    return jsonify(response)

def main():
    return app.run()

main()