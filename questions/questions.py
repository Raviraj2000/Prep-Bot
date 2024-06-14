import random

def get_interview_question():

    interview_questions = [
        "Tell me about yourself.",
        "Why do you want to work here?",
        "What are your greatest strengths?",
        "What are your weaknesses?",
        "Where do you see yourself in five years?",
        "Describe a time when you had to overcome a significant challenge at work.",
        "Why are you leaving your current job?",
        "Tell me about a time when you worked as part of a team.",
        "Describe a situation where you had to work independently.",
        "How do you handle stress and pressure?",
        "What is your greatest professional achievement?",
        "Tell me about a time when you failed.",
        "How do you prioritize your work?",
        "Why should we hire you?",
        "Describe a time when you had to manage a conflict at work.",
        "What motivates you?",
        "How do you handle criticism?",
        "What are your salary expectations?",
        "What do you know about our company?",
        "Do you have any questions for us?"
    ]

    return random.choice(interview_questions)