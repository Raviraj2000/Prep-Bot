from groq import Groq
import os
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)
def evaluate(question,retrieved_steps, candidate_answer, resume_info):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages = [
            {
            "role": "system",
            "content": f"You are an AI expert in job interview preparation. Your task is to help job seekers improve their interview answers.\n\n"
                    f"Given the following inputs:\n\n"
                    f"Question: \"{question}\"\n"
                    f"Retrieved Steps: \"{retrieved_steps}\"\n"
                    f"Candidate's Answer: \"{candidate_answer}\"\n\n"
                    f"Candidate's Resume Information: \"{resume_info}\"\n\n"
                    f"Analyze the candidate’s answer, referencing the retrieved steps only if they are relevant to the question. Provide detailed feedback on how to improve the response, and ignore any steps or sample answers that do not apply. Specifically, highlight:\n\n"
                    f"- **Strengths:** Identify and explain the good parts of the candidate's answer.\n"
                    f"- **Areas for Improvement:** Identify and explain the parts of the candidate's answer that need improvement.\n"
                    f"- **Suggestions for Improvement:** Provide specific suggestions on how to improve the answer, including examples of what to say instead.\n"
                    f"  - If the candidate's resume contains relevant information, incorporate it into the suggestions.\n"
                    f"  - Highlight specific skills, experiences, or achievements from the resume that could strengthen the candidate's answer.\n\n"
                    f"Here is an example of the feedback format and content:\n\n"
                    f"**Question:** \"Why do you want to work here?\"\n"
                    f"**Candidate's Answer:** \"I have heard good things about your company.\"\n"
                    f"**Strengths:**\n"
                    f"1. The candidate has a positive view of the company.\n"
                    f"**Areas for Improvement:**\n"
                    f"1. The answer lacks specifics about the company.\n"
                    f"2. The candidate does not mention how their career goals align with the company's mission.\n"
                    f"**Suggestions for Improvement:**\n"
                    f"1. Consider saying: 'I am impressed by your company's commitment to innovation and excellence.'\n"
                    f"2. Explain how your career goals and values align with the company’s mission and values.\n"
                    f"3. Referencing your experience as a 'Lead Software Engineer at Easley Dunn Productions', highlight your experience in cross-functional collaboration and problem-solving skills.\n\n"
                    f"Here is a JSON object example:\n"
                    f"{{\n"
                    f"  \"Question\": \"Describe a situation where you had to work independently.\",\n"
                    f"  \"Candidate's Answer\": \"I like to work in a team\",\n"
                    f"  \"Feedback\": {{\n"
                    f"    \"Strengths\": [\"No feedback\"],\n"
                    f"    \"Areas for Improvement\": [\n"
                    f"      \"The answer does not address the question.\",\n"
                    f"      \"The answer is contradictory, as the question asks about working independently, but the candidate mentions they like to work in a team.\"\n"
                    f"    ],\n"
                    f"    \"Suggestions for Improvement\": [\n"
                    f"      \"Consider telling a specific story of a time when you had to work independently, such as a project or task where you didn't have direct supervision.\",\n"
                    f"      \"For example, referencing your experience as a 'Database Engineer at USC Facilities Planning and Management', you could describe how you independently managed data migration tasks and improved report generation systems.\"\n"
                    f"    ]\n"
                    f"  }}\n"
                    f"}}\n\n"
                    f"Now, provide feedback based on the inputs in JSON:\n\n"
                    f"{{\n"
                    f"  \"Feedback\": {{\n"
                    f"    \"Strengths\": [\"No feedback\"],\n"
                    f"    \"Areas for Improvement\": [\"No feedback\"],\n"
                    f"    \"Suggestions for Improvement\": [\"No feedback\"]\n"
                    f"  }}\n"
                    f"}}"
                }
        ],
        temperature=0.8,
        max_tokens=4096,
        top_p=0.9,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    return completion.choices[0].message.content

def parse_resume(text: str):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": f"You are an AI assistant specialized in analyzing resumes.\n"
                    f"Given the following resume below:\n\n"
                    f"Resume: \"{text}\"\n"
                    f"Extract the following key details from the given resume text:\n"
                    f"- Name\n"
                    f"- Education\n"
                    f"- Experience\n"
                    f"- Projects\n"
                    f"- Certifications\n"
                    f"- Hobbies\n\n"
                    f"Provide the extracted information in the following JSON format:\n"
                    f"{{\n"
                    f"  \"Name\": \"\",\n"
                    f"  \"Education\": \"\",\n"
                    f"  \"Experience\": \"\",\n"
                    f"  \"Projects\": \"\",\n"
                    f"  \"Certifications\": \"\",\n"
                    f"  \"Hobbies\": \"\"\n"
                    f"}}\n"
                    "If any section is missing in the resume, leave the field empty."
                
            }
        ],
        temperature=1,
        max_tokens=4096,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    return completion.choices[0].message.content