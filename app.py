from transcription.whisper import processor, model
import streamlit as st
import requests
import json
from transcription.whisper_transcribe import get_transcription

st.title("💬 Interview Bot")
st.caption("🚀 A Streamlit chatbot powered by Llama")

def generate_report(info, type):
    st.subheader(type)
    if not info:
        st.markdown("This answer has no ", type)
    else:
        for s in info:
            st.markdown(s)
    return

if "messages" not in st.session_state:
   question = requests.get("http://127.0.0.1:5000/api/question")
   string_data = question.content.decode('utf-8')
   st.session_state["messages"] = [{"role": "assistant", "content": string_data}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.button("Record Answer"):
    
    answer = get_transcription(processor, model)
    with st.chat_message("user"):
        st.markdown(answer)

    data = {"question":st.session_state.messages[-1]['content'], 'candidate_answer' : answer}
    response = requests.post("http://127.0.0.1:5000/api/evaluate", data=data)
    feedback = json.loads(response.content)
    feedback = feedback['Feedback']
    print(feedback)

    strengths = feedback['Strengths']
    afi = feedback['Areas for Improvement']
    sfi = feedback['Suggestions for Improvement']
    
    st.header("Report")
    generate_report(strengths, "Strengths")
    generate_report(afi, "Areas for Improvement")
    generate_report(sfi, "Suggestions for Improvement")