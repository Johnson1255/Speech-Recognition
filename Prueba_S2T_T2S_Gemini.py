import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import google.generativeai as genai
import textwrap

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
max_chars = 50  
texto1 = textwrap.shorten(response1.text, max_chars)

texto1 = response1.text

tts = gTTS(text=texto1, lang="es")
filename = "audiot2s.mp3"
tts.save(filename)

playsound(filename)