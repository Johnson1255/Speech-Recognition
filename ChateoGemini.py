import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
from gtts import gTTS
import os

# Carga del modelo pre-entrenado Wav2Vec2
modelo = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
procesador = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Función para transcribir audio a texto
def transcribir_audio(ruta_audio):
    try:
        # Carga y preprocesamiento del audio
        audio, sample_rate = librosa.load(ruta_audio, sr=16000)
        input_values = procesador(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        
        # Inferencia del modelo
        with torch.no_grad():
            predicciones = modelo(input_values.input_values)
        
        # Decodificación de las predicciones
        transcripcion = procesador.batch_decode(predicciones.logits.argmax(dim=-1))[0]
        
        return transcripcion
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_audio}' no se encontró.")
        return None
    except Exception as e:
        print(f"Se produjo un error al transcribir el audio: {e}")
        return None

# Función para convertir texto a audio utilizando gTTS
def texto_a_audio(texto, nombre_archivo="output.mp3"):
    tts = gTTS(texto, lang='es')
    tts.save(nombre_archivo)
    return nombre_archivo

# Función para reproducir el audio generado
def reproducir_audio(ruta_audio):
    os.system(f"start {ruta_audio}")  # Esto funcionará en Windows, usa un comando apropiado para tu sistema operativo

# Ejemplo de uso
ruta_audio = "audio.mp3"  # Asegúrate de que este archivo exista en la ruta especificada
texto_transcrito = transcribir_audio(ruta_audio)
if texto_transcrito:
    print(f"Transcripción: {texto_transcrito}")

    texto_a_convertir = "Hola, ¿cómo estás?"
    ruta_audio_generado = texto_a_audio(texto_a_convertir)
    reproducir_audio(ruta_audio_generado)
else:
    print("No se pudo transcribir el audio.")
