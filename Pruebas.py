import openai
import speech_recognition as sr
import pyttsx3
import time

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
    max_reintentos = 5
    reintentos = 0
    while reintentos < max_reintentos:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente útil."},
                    {"role": "user", "content": texto}
                ]
            )
            respuesta = response['choices'][0]['message']['content']
            return respuesta
        except openai.error.RateLimitError:
            print("Límite de tasa excedido. Esperando 5 segundos antes de reintentar...")
            time.sleep(5)
            reintentos += 1
        except openai.error.OpenAIError as e:
            print(f"Ocurrió un error: {e}")
            return "Lo siento, ocurrió un error al generar la respuesta."
    print("Se ha excedido el número máximo de reintentos.")
    return "Lo siento, no puedo procesar tu solicitud en este momento."

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
