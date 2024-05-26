from gtts import gTTS
from playsound import playsound

texto = "El día de hoy es sabado 25 de mayo de 2024"

tts = gTTS(text=texto, lang="es")
filename = "audio.mp3"
tts.save(filename)

playsound(filename)
