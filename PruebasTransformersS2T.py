import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd
import numpy as np

# Carga del modelo pre-entrenado Wav2Vec2
modelo_wav2vec2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
procesador_wav2vec2 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

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

        return transcripcion
    except Exception as e:
        print(f"Se produjo un error al transcribir el audio desde el micrófono: {e}")
        return None

# Ejemplo de uso con audio desde el micrófono
texto_transcrito = transcribir_audio_microfono()
if texto_transcrito:
    print(f"Transcripción Wav2Vec2 desde el micrófono: {texto_transcrito}")
else:
    print("No se pudo transcribir el audio desde el micrófono.")