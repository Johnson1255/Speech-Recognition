import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import language_tool_python
from gtts import gTTS
from playsound import playsound
from flask import Flask, request, jsonify
import soundfile as sf
import google.generativeai as genai
import textwrap
import re

# Carga del modelo pre-entrenado Wav2Vec2
modelo_wav2vec2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")
procesador_wav2vec2 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")

# Inicializa el corrector gramatical de LanguageTool para español
tool = language_tool_python.LanguageTool('es')

# Configura la API de Google Generative AI
genai.configure(api_key='')

model = genai.GenerativeModel('gemini-pro')

app = Flask(__name__)

def transcribir_audio(audio, samplerate):
    try:
        # Preprocesamiento del audio para Wav2Vec2
        audio = np.squeeze(audio)
        input_values = procesador_wav2vec2(audio, sampling_rate=samplerate, return_tensors="pt", padding=True)

        # Inferencia del modelo Wav2Vec2
        with torch.no_grad():
            logits = modelo_wav2vec2(input_values.input_values).logits

        # Decodificación de las predicciones
        transcripcion = procesador_wav2vec2.batch_decode(logits.argmax(dim=-1))[0]
        transcripcion_corregida = tool.correct(transcripcion)

        return transcripcion_corregida
    
    except Exception as e:
        return str(e)

@app.route('/transcribir', methods=['POST'])
def transcribir():
    if 'audio' not in request.files:
        return jsonify({"error": "No se ha proporcionado ningún archivo de audio"}), 400
    
    audio_file = request.files['audio']
    audio, samplerate = sf.read(audio_file)
    
    try:
        transcripcion_corregida = transcribir_audio(audio, samplerate)
        
        # Generación de contenido con la API de Google Generative AI
        response1 = model.generate_content(transcripcion_corregida)
        max_chars = 500
        texto1 = textwrap.shorten(response1.text, max_chars)
        texto_limpio = re.sub(r"\*", "", texto1).strip()

        # Convertir la respuesta a audio
        tts = gTTS(text=texto_limpio, lang="es")
        filename = "audio_respuesta_D.mp3"
        tts.save(filename)

        # Reproducir el audio generado
        playsound(filename)

        # Devolver la respuesta transcrita y la URL del audio generado
        return jsonify({"transcripcion": transcripcion_corregida, "respuesta": texto_limpio, "audio_url": filename})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
