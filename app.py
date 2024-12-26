from transcription.whisper import processor, model
import streamlit as st
import requests
import json
from transcription.whisper_transcribe import get_transcription
import sounddevice as sd
import numpy as np

# --- Utility Functions ---
def generate_report(info, type):
    if not info:
        st.markdown(f"No data available for {type}")
    else:
        for s in info:
            st.markdown(s)
    return

def fetch_new_question():
    try:
        response = requests.get("http://127.0.0.1:5000/api/question")
        if response.status_code == 200:
            return response.content.decode('utf-8')
        else:
            return "Failed to fetch a new question. Please try again."
    except Exception as e:
        return f"An error occurred: {e}"

def evaluate_answer(question, answer, candidate_name):
    data = {
        "question": question,
        "candidate_answer": answer,
        "candidate_name": candidate_name
    }
    try:
        response = requests.post("http://127.0.0.1:5000/api/evaluate", data=data)
        if response.status_code == 200:
            feedback = json.loads(response.content)['Feedback']
            return feedback
        else:
            return {"Strengths": [], "Areas for Improvement": [], "Suggestions for Improvement": []}
    except Exception as e:
        return {"Strengths": [], "Areas for Improvement": [], "Suggestions for Improvement": [], "Error": str(e)}

# Recording Helper Functions
def start_recording():
    st.session_state["is_recording"] = True
    st.toast("Recording started. Speak into the microphone...")

    audio_data = []  # Local list to store audio data

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())

    st.session_state["stream"] = sd.InputStream(
        samplerate=16000, channels=1, dtype='float32', callback=callback
    )
    st.session_state["audio_data"] = audio_data  # Safe to store initial list
    st.session_state["stream"].start()


def stop_recording():
    if "stream" in st.session_state and st.session_state["stream"] is not None:
        st.session_state["stream"].stop()
        st.session_state["stream"].close()
        st.session_state["is_recording"] = False
        st.toast("Recording stopped. Processing audio...")
        st.session_state["final_audio_data"] = np.concatenate(st.session_state["audio_data"], axis=0)
        st.session_state.pop("audio_data")  # Clean up temporary storage


# --- Sidebar: User Details ---
with st.sidebar:
    st.header("📂 User Details")
    user_name = st.text_input("Enter your name")
    uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=['pdf'])
    
    if uploaded_file:
        if st.button("Submit Resume") and user_name:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")}
                data = {"name": user_name}
                response = requests.post("http://127.0.0.1:5000/api/upload_resume", files=files, data=data)
                if response.status_code == 200:
                    st.toast("Resume analyzed successfully!")
                else:
                    st.error("Failed to analyze resume. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Main Chat Layout (Single Column) ---
st.title("💬 Interview Bot")
st.caption("🚀 A Streamlit chatbot powered by Llama")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "fetch_next_question" not in st.session_state:
    st.session_state.fetch_next_question = False

# --- Section 1: Question ---
st.header("📜 Question")

# Fetch a new question only when fetch_next_question is True
if st.session_state.fetch_next_question:
    question = fetch_new_question()
    st.session_state.messages.append({"role": "assistant", "content": question})
    st.session_state.fetch_next_question = False  # Reset flag

# Filter only assistant (bot) messages
bot_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]

if bot_messages:
    st.info(bot_messages[-1]["content"])
else:
    st.info("No question available. Click 'Next Question' to start.")

# --- Section 2: Transcribed Answer ---
st.header("🎙️ Transcribed Answer")
if st.session_state.answers:
    st.success(st.session_state.answers[-1])  # Show the most recent answer
else:
    st.info("No answer recorded yet. Click 'Start Recording' to begin.")

# --- Section 3: Feedback ---
st.header("📊 Feedback")
if st.session_state.feedback:
    feedback = st.session_state.feedback[-1]
    strengths = feedback.get('Strengths', [])
    afi = feedback.get('Areas for Improvement', [])
    sfi = feedback.get('Suggestions for Improvement', [])

    st.subheader("✅ Strengths")
    generate_report(strengths, "Strengths")

    st.subheader("⚠️ Areas for Improvement")
    generate_report(afi, "Areas for Improvement")

    st.subheader("💡 Suggestions for Improvement")
    generate_report(sfi, "Suggestions for Improvement")
else:
    st.info("No feedback available yet. Please record an answer to see feedback.")

# --- Bottom Buttons Section ---
st.markdown("---")

if st.button("Next Question"):
    st.session_state.fetch_next_question = True
    st.rerun()

if not st.session_state.is_recording and st.button("🎙️ Start Recording"):
    start_recording()
    st.rerun()

if st.session_state.is_recording and st.button("⏹️ Stop Recording"):
    stop_recording()
    audio = st.session_state["final_audio_data"]
    answer = get_transcription(processor, model, audio, 16000)
     
    st.session_state.messages.append({"role": "user", "content": answer})
    st.session_state.answers.append(answer)

    feedback = evaluate_answer(
        st.session_state.messages[-1]['content'],
        answer,
        user_name
    )
    st.session_state.feedback.append(feedback)
    st.rerun()
