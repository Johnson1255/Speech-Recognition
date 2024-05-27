"""
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

inputs = processor(text="HOLA COMO HAS ESTADO?", return_tensors="pt") #La voz le sale muy extrajero

from datasets import load_dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

import torch
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

from transformers import SpeechT5HifiGan
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

with torch.no_grad():
    speech = vocoder(spectrogram)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

import soundfile as sf
sf.write("tts_example.wav", speech.numpy(), samplerate=16000)
"""

from transformers import pipeline
import scipy

# Initialize the Text-to-Speech pipeline with the Bark model
synthesiser = pipeline("text-to-speech", "suno/bark")

# Define the text you want to convert to speech
text = "Hello, my dog is cooler than you!"

# Generate speech using the pipeline with optional sampling for variation
speech = synthesiser(text, forward_params={"do_sample": True})

# Save the generated speech to a WAV file using scipy
scipy.io.wavfile.write("bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])

print("Speech saved to bark_out.wav")