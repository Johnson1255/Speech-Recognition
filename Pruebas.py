import openai
import speech_recognition as sr
import pyttsx3

# Configuración de la API de OpenAI
openai.api_key = ''

def transcribir_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Escuchando...")
        audio = recognizer.listen(source)
        try:
            texto = recognizer.recognize_google(audio, language='es-ES')
            print("Tú dijiste: " + texto)
            return texto
        except sr.UnknownValueError:
            print("Lo siento, no entendí lo que dijiste.")
            return ""
        except sr.RequestError:
            print("Error al conectarse al servicio de reconocimiento de voz.")
            return ""

def generar_respuesta(texto):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": texto}
        ]
    )
    respuesta = response['choices'][0]['message']['content']
    return respuesta

def hablar(texto):
    engine = pyttsx3.init()
    engine.setProperty('voice', 'spanish')
    engine.say(texto)
    engine.runAndWait()

if __name__ == "__main__":
    while True:
        texto_usuario = transcribir_audio()
        if texto_usuario:
            respuesta_ia = generar_respuesta(texto_usuario)
            print("Asistente: " + respuesta_ia)
            hablar(respuesta_ia)
