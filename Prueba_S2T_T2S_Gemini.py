import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import google.generativeai as genai
import textwrap
import re

r = sr.Recognizer()
mic = sr.Microphone()

#Imprimir un "Escuchando" para saber que el sistema puede empezar a reconocer la voz
print("Escuchando...")
with mic as source:
    audio = r.listen(source)

try:
    texto = r.recognize_google(audio)
    print(f"Texto: {texto}")

    # Manejo de error si la voz no se detecta de manera adecuada
except sr.UnknownValueError:
    print("Error: No se pudo entender el audio.")

    # Error Inesperado x
except sr.RequestError as e:
    print(f"Error: {e}")

genai.configure(api_key='')
model = genai.GenerativeModel('gemini-pro')

PROMPT = {texto}
response1 = model.generate_content(PROMPT)
max_chars = 500
texto1 = textwrap.shorten(response1.text, max_chars)

print("Con *: " + texto1)

texto_limpio = re.sub(r"\*", "", texto1)
texto_limpio = texto_limpio.strip()

print("Sin *: " + texto_limpio)

tts = gTTS(text=texto_limpio, lang="es")
filename = "audio_t2s.mp3"
tts.save(filename)

playsound(filename)