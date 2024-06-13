from whisper import processor, model
import streamlit as st
import requests
import numpy as np
import torch
import torchaudio
import keyboard
import sounddevice as sd
import json

st.title("💬 Interview Bot")
st.caption("🚀 A Streamlit chatbot powered by Llama")

# Function to preprocess audio for Whisper
def preprocess_audio(audio, sample_rate):
    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio = resampler(torch.tensor(audio).transpose(0, 1)).transpose(0, 1).numpy()
    return audio

def record_audio():
    sample_rate = 16000  # Whisper expects 16000Hz input
    audio_data = []

    def callback(indata, frames, time, status):
        audio_data.append(indata.copy())

    with st.chat_message("assistant"):
        st.markdown("Press 'R' to start recording your answer and 'R' again to stop.")
    keyboard.wait('r')
    with st.chat_message("assistant"):
        st.markdown("Recording started please speak into the microphone... Press 'R' again to stop.")
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback, device=48)
    stream.start()
    keyboard.wait('r')
    stream.stop()
    stream.close()
    with st.chat_message("assistant"):
        st.markdown("Recording stopped.")

    audio = np.concatenate(audio_data, axis=0)
    return audio, sample_rate


def get_transcription():
    audio, sample_rate = record_audio()

    waveform = preprocess_audio(audio, sample_rate)

    # Prepare inputs for the model
    inputs = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000)

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])

    # Decode the predicted tokens
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

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
    # if msg["role"] == "user":
    #     continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.button("Record Answer"):
    
    answer = get_transcription()
    with st.chat_message("user"):
        st.markdown(answer)

    data = {"question":st.session_state.messages[-1]['content'], 'candidate_answer' : answer}
    response = requests.post("http://127.0.0.1:5000/api/evaluate", data=data)

    feedback_str = json.loads(response.content)
    feedback = json.loads(feedback_str)

    feedback = feedback['Feedback']
    print(feedback)

    strengths = []
    afi = []
    sfi = []

    if 'Strengths' in feedback:
        strengths = feedback['Strengths']

    if 'Areas for Improvement' in feedback:
        afi = feedback['Areas for Improvement']

    if 'Suggestions for Improvement' in feedback:
        sfi = feedback['Suggestions for Improvement']

    st.header("Report")
    generate_report(strengths, "Strengths")
    generate_report(afi, "Areas for Improvement")
    generate_report(sfi, "Suggestions for Improvement")
    
