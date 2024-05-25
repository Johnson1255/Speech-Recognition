import speech_recognition as sr
import pyttsx3
import os

from dotenv import load_dotenv
load_dotenv()
OPENAI_KEY = os.getenv('')

import openai
openai.api_key = OPENAI_KEY

#Hablar
def SpeakText(command):
    engine = pytysx3.init()
    engine.say(command)
    engine.runAndWait()

r = sr.Recognizer()

#Grabar audio
def record_text():
    while(1):
        try:
            with sr.Microphone() as source2:

                r.adjust_for_ambient_noise(source2, duration = 0.2)
                print("Estoy escuchandote")

                audio2 = r.listen(source2)

                MyText = r.recognize_google(audio2)

                return MyText
        except sr.RequestError as e:
            print("No hubieron resultados; {0}".format(e))

        except sr.UnknownValueError:
            print("Ocurrio un error desconocido")