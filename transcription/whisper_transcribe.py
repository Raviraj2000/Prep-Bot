import torch
import torchaudio
import numpy as np

# Function to record audio from the microphone
def preprocess_audio(audio, sample_rate):
    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio = resampler(torch.tensor(audio).transpose(0, 1)).transpose(0, 1).numpy()
    return audio


def get_transcription(processor, model, audio, sample_rate):
    # audio, sample_rate = record_audio()

    waveform = preprocess_audio(audio, sample_rate)

    # Prepare inputs for the model
    inputs = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000)

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])

    # Decode the predicted tokens
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription
