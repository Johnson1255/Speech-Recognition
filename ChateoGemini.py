import torch
from transformers import AutoModelForCTC, Wav2Vec2Processor

# Carga del modelo pre-entrenado Wav2Vec2
modelo = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base")
procesador = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Función para transcribir audio a texto
def transcribir_audio(ruta_audio):
  # Carga y preprocesamiento del audio
  audio, sample_rate = librosa.load(ruta_audio, sr=16000)
  input_values = procesador(audio, sample_rate=sample_rate, return_tensors="pt")

  # Inferencia del modelo
  with torch.no_grad():
    predicciones = modelo(**input_values)

  # Decodificación de las predicciones
  transcripcion = procesador.decode_logits(predicciones.logits.transpose(0, 2))[0]

  return transcripcion

# Función para convertir texto a audio (requiere librería adicional como TTS)
def texto_a_audio(texto):
  # Conversión de texto a representación de audio
  # (Utilizar librería TTS como gTTS o pyttsx3)
  audio_representacion = convertir_texto_a_audio(texto)

  # Generación del audio
  audio = generar_audio_desde_representacion(audio_representacion)

  return audio

# Ejemplo de uso
ruta_audio = "audio.wav"
texto_transcrito = transcribir_audio(ruta_audio)
print(f"Transcripción: {texto_transcrito}")

texto_a_convertir = "Hola, ¿cómo estás?"
audio_generado = texto_a_audio(texto_a_convertir)
reproducir_audio(audio_generado)
