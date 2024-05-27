import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd
import numpy as np
import language_tool_python
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import google.generativeai as genai
import textwrap
import re

# Carga del modelo pre-entrenado Wav2Vec2
modelo_wav2vec2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")
procesador_wav2vec2 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")

# Inicializa el corrector gramatical de LanguageTool para español
tool = language_tool_python.LanguageTool('es')

# Función para transcribir audio desde el micrófono
def transcribir_audio_microfono():
    try:
        # Configuración de la grabación de audio desde el micrófono
        fs = 16000  # Frecuencia de muestreo
        duracion = 5  # Duración de la grabación en segundos

        print("Grabando audio...")

        # Captura de audio desde el micrófono
        audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()  # Espera a que termine la grabación

        print("Grabación finalizada.")

        # Preprocesamiento del audio para Wav2Vec2
        audio = np.squeeze(audio)
        input_values = procesador_wav2vec2(audio, sampling_rate=fs, return_tensors="pt", padding=True)

        # Inferencia del modelo Wav2Vec2
        with torch.no_grad():
            logits = modelo_wav2vec2(input_values.input_values).logits

        # Decodificación de las predicciones
        transcripcion = procesador_wav2vec2.batch_decode(logits.argmax(dim=-1))[0]
        print("Sin corregir: " + transcripcion)

        transcripcion_corregida = tool.correct(transcripcion)
        print("\nCorregida: " + transcripcion_corregida)

        return transcripcion_corregida
    
    except Exception as e:
        print(f"Se produjo un error al transcribir el audio desde el micrófono: {e}")
        return None

# Ejemplo de uso con audio desde el micrófono

genai.configure(api_key='')
model = genai.GenerativeModel('gemini-pro')

texto_transcrito = transcribir_audio_microfono()

PROMPT = texto_transcrito
response1 = model.generate_content(PROMPT)
max_chars = 500
texto1 = textwrap.shorten(response1.text, max_chars)

print("Con *: " + texto1)

texto_limpio = re.sub(r"\*", "", texto1)
texto_limpio = texto_limpio.strip()

print("\nSin *: " + texto_limpio)

tts = gTTS(text=texto_limpio, lang="es")
filename = "audio_respuesta.mp3"
tts.save(filename)

playsound(filename)