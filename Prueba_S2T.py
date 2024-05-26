import speech_recognition as sr

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
