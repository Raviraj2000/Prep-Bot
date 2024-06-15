import torch
import torchaudio
import sounddevice as sd
import numpy as np
import keyboard
import streamlit as st


# Function to record audio from the microphone
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
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback, device=sd.default.device[0])
    stream.start()
    keyboard.wait('r')
    stream.stop()
    stream.close()
    with st.chat_message("assistant"):
        st.markdown("Recording stopped.")

    audio = np.concatenate(audio_data, axis=0)
    return audio, sample_rate


def get_transcription(processor, model):
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
