from flask import Flask, request, jsonify
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from googletrans import Translator

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Inicializar traductor
translator = Translator()

# Cargar modelo y vectorizador al iniciar la aplicación
try:
    modelo_emocion = joblib.load("modelo_emocional.pkl")
    vectorizer = joblib.load("vectorizador_emocional.pkl")
    logger.info("Modelo y vectorizador cargados exitosamente")
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")
    modelo_emocion = None
    vectorizer = None

# Diccionario para traducir emociones al español
EMOTION_TRANSLATIONS = {
    'angry': 'enojado',
    'annoyed': 'molesto',
    'anticipating': 'expectante',
    'anxious': 'ansioso',
    'apprehensive': 'aprensivo',
    'ashamed': 'avergonzado',
    'caring': 'cariñoso',
    'confident': 'confiado',
    'content': 'contento',
    'devastated': 'devastado',
    'disappointed': 'decepcionado',
    'disgusted': 'disgustado',
    'embarrassed': 'avergonzado',
    'excited': 'emocionado',
    'faithful': 'fiel',
    'furious': 'furioso',
    'grateful': 'agradecido',
    'guilty': 'culpable',
    'hopeful': 'esperanzado',
    'impressed': 'impresionado',
    'jealous': 'celoso',
    'joyful': 'alegre',
    'lonely': 'solitario',
    'nostalgic': 'nostálgico',
    'prepared': 'preparado',
    'proud': 'orgulloso',
    'sad': 'triste',
    'sentimental': 'sentimental',
    'surprised': 'sorprendido',
    'terrified': 'aterrorizado',
    'trusting': 'confiado'
}

EMOTION_EMOJIS = {
    'alegre': '😊', 'triste': '😢', 'enojado': '😠', 'emocionado': '🤩',
    'ansioso': '😰', 'cariñoso': '❤️', 'confiado': '😎', 'decepcionado': '😞',
    'esperanzado': '🌟', 'avergonzado': '😳', 'celoso': '😒', 'nostálgico': '📸',
    'orgulloso': '🎖️', 'culpable': '😓', 'sorprendido': '😮'
}



def detect_language_and_translate(text):
    """
    Detecta el idioma y traduce al inglés si es necesario
    """
    try:
        # Detectar idioma
        detection = translator.detect(text)
        detected_lang = detection.lang
        
        logger.info(f"Idioma detectado: {detected_lang}")
        
        # Si no es inglés, traducir
        if detected_lang != 'en':
            translated = translator.translate(text, src=detected_lang, dest='en')
            english_text = translated.text
            logger.info(f"Texto original: {text}")
            logger.info(f"Texto traducido: {english_text}")
            return english_text, detected_lang
        else:
            return text, 'en'
            
    except Exception as e:
        logger.error(f"Error en traducción: {e}")
        # Si falla la traducción, asumir que está en inglés
        return text, 'en'

@app.route('/')
def home():
    return render_template('index.html')

# Ruta para archivos estáticos (CSS/JS)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    try:
        if modelo_emocion is None or vectorizer is None:
            return jsonify({"error": "Modelo no disponible"}), 500

        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Se requiere el campo 'text'"}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({"error": "El texto no puede estar vacío"}), 400

        # Procesamiento directo (sin traducción)
        text_vectorized = vectorizer.transform([text])
        emotion_pred = modelo_emocion.predict(text_vectorized)[0]
        probabilities = modelo_emocion.predict_proba(text_vectorized)[0]
        confidence = float(np.max(probabilities))

        # Traducir emoción y añadir emoji
        emotion_spanish = EMOTION_TRANSLATIONS.get(emotion_pred, emotion_pred)
        emoji = EMOTION_EMOJIS.get(emotion_spanish, '😐')
        
        return jsonify({
            "emotion": emotion_pred,  # Devuelve la emoción en inglés directamente
            "emotion_spanish": emotion_spanish,
            "emoji": emoji,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    """
    Endpoint para probar el modelo con ejemplos
    """
    try:
        if modelo_emocion is None or vectorizer is None:
            return jsonify({"error": "Modelo no disponible"}), 500
        
        # Ejemplos de prueba
        test_cases = [
            "I feel sad and tired",
            "I am very happy today",
            "This makes me angry",
            "I love this so much"
        ]
        
        results = []
        for text in test_cases:
            vectorized = vectorizer.transform([text])
            pred = modelo_emocion.predict(vectorized)[0]
            prob = modelo_emocion.predict_proba(vectorized)[0]
            confidence = float(np.max(prob))
            
            results.append({
                "input": text,
                "emotion_english": pred,
                "emotion_spanish": EMOTION_TRANSLATIONS.get(pred, pred),
                "emoji": EMOTION_EMOJIS.get(EMOTION_TRANSLATIONS.get(pred, pred), '😐'),
                "confidence": round(confidence, 3)
            })
        
        return jsonify({
            "test_results": results,
            "model_status": "working"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/emotions-list', methods=['GET'])
def get_emotions_list():
    """
    Retorna lista completa de emociones con emojis
    """
    emotions = []
    for en, es in EMOTION_TRANSLATIONS.items():
        emotions.append({
            "english": en,
            "spanish": es,
            "emoji": EMOTION_EMOJIS.get(es, '😐')
        })
    
    return jsonify({
        "emotions": emotions,
        "total": len(emotions),
        "supported_languages": "Cualquier idioma (traducción automática)"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Verificar estado completo de la API
    """
    try:
        health_status = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Verificar modelo
        if modelo_emocion is not None:
            health_status["components"]["model"] = "✅ loaded"
        else:
            health_status["components"]["model"] = "❌ not loaded"
        
        # Verificar vectorizador
        if vectorizer is not None:
            health_status["components"]["vectorizer"] = "✅ loaded"
        else:
            health_status["components"]["vectorizer"] = "❌ not loaded"
        
        # Verificar traductor
        try:
            test_translation = translator.translate("test", dest='es')
            health_status["components"]["translator"] = "✅ working"
        except:
            health_status["components"]["translator"] = "❌ not working"
        
        # Prueba completa del pipeline
        if modelo_emocion is not None and vectorizer is not None:
            try:
                test_vectorized = vectorizer.transform(["I am happy"])
                test_pred = modelo_emocion.predict(test_vectorized)[0]
                health_status["test_prediction"] = {
                    "input": "I am happy",
                    "emotion_english": test_pred,
                    "emotion_spanish": EMOTION_TRANSLATIONS.get(test_pred, test_pred)
                }
                health_status["status"] = "healthy"
            except Exception as e:
                health_status["status"] = "error"
                health_status["error"] = str(e)
        else:
            health_status["status"] = "unhealthy"

        return jsonify(health_status)

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("🎭 Detector de Emociones - API v2.0")
    print("=" * 50)
    print("📊 Modelo:", "✅ Cargado" if modelo_emocion is not None else "❌ Error")
    print("🔤 Vectorizador:", "✅ Cargado" if vectorizer is not None else "❌ Error")
    print("🌍 Traductor:", "✅ Disponible")
    print("=" * 50)
    print("🌐 Endpoints disponibles:")
    print("   - GET  /              -> Información de la API")
    print("   - POST /detect-emotion -> Detectar emoción (cualquier idioma)")
    print("   - GET  /test-model     -> Probar modelo con ejemplos")
    print("   - GET  /emotions-list  -> Lista completa de emociones")
    print("   - GET  /health         -> Estado detallado del sistema")
    print("=" * 50)
    print("🚀 Servidor iniciando en: http://localhost:5000")
    print("💡 Acepta texto en cualquier idioma!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)