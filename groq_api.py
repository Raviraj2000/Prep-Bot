from groq import Groq
client = Groq()
def evaluate(question,retrieved_steps, candidate_answer):
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
                   f"Analyze the candidate's answer based on the retrieved steps and provide detailed feedback on how to improve their response. Specifically, highlight:\n\n"
                   f"- Strengths: Identify and explain the good parts of the candidate's answer.\n"
                   f"- Areas for Improvement: Identify and explain the parts of the candidate's answer that need improvement.\n"
                   f"- Suggestions for Improvement: Provide specific suggestions on how to improve the answer, including examples of what to say instead.\n\n"
                   f"Here is an example of the feedback format and content:\n\n"
                   f"Question: \"Why do you want to work here?\"\n"
                   f"Candidate's Answer: \"I have heard good things about your company.\"\n"
                   f"Strengths:\n1. The candidate has a positive view of the company.\n"
                   f"Areas for Improvement:\n1. The answer lacks specifics about the company.\n"
                   f"2. The candidate does not mention how their career goals align with the company's mission.\n"
                   f"Suggestions for Improvement:\n1. Consider saying: 'I am impressed by your company's commitment to innovation and excellence.'\n"
                   f"2. Explain how your career goals and values align with the company’s mission and values.\n\n"
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
                   f"      \"For example, you could say: 'In my previous role, I was tasked with creating a marketing campaign for a new product launch. I had to work independently to research the target audience, develop a campaign strategy, and design the marketing materials.'\"\n"
                   f"    ]\n"
                   f"  }}\n"
                   f"}}\n\n"
                   f"Now, provide feedback based on the inputs in JSON:\n\n"
                   f"{{\n"
                   f"  \"Question\": \"{question}\",\n"
                   f"  \"Candidate's Answer\": \"{candidate_answer}\",\n"
                   f"  \"Feedback\": {{\n"
                   f"    \"Strengths\": [\"No feedback\"],\n"
                   f"    \"Areas for Improvement\": [\"No feedback\"],\n"
                   f"    \"Suggestions for Improvement\": [\"No feedback\"]\n"
                   f"  }}\n"
                   f"}}"
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
