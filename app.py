import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.models import Model
import joblib
from flask import Flask, request, jsonify, render_template
from deep_translator import GoogleTranslator
import google.generativeai as genai
import os
import requests
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__, static_folder="static")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = os.getenv("GEMINI_URL")

# Cargar tokenizadores
inp_tokenizer = joblib.load(open("inp_tokenizer.pickle", "rb"))
targ_tokenizer = joblib.load(open("targ_tokenizer.pickle", "rb"))

# Configuración
config = {
    "vocab_inp_size": len(inp_tokenizer.word_index) + 1,
    "vocab_tar_size": len(targ_tokenizer.word_index) + 1,
    "embedding_dim": 256,
    "units": 512,
    "BATCH_SIZE": 1,
    "max_length_inp": 20,
    "max_length_targ": 20
}

# Definición de Encoder, Attention, Decoder...
class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

# Cargar el modelo
encoder = Encoder(config["vocab_inp_size"], config["embedding_dim"], config["units"], config["BATCH_SIZE"])
decoder = Decoder(config["vocab_tar_size"], config["embedding_dim"], config["units"], config["BATCH_SIZE"])
encoder.build(input_shape=(config["BATCH_SIZE"], config["max_length_inp"]))
decoder.build(input_shape=[(config["BATCH_SIZE"], 1), (config["BATCH_SIZE"], config["units"]), (config["BATCH_SIZE"], config["max_length_inp"], config["units"])])
encoder.load_weights("chatbot_encoder.weights.h5")
decoder.load_weights("chatbot_decoder.weights.h5")

class ChatbotModel:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

modelo_chatbot = ChatbotModel(encoder, decoder)

# Cargar modelo de emociones
modelo_emociones = joblib.load("modelo_emocional.pkl")
vectorizer = joblib.load("vectorizador_emocional.pkl")

# Rutas
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    mensaje = data.get("mensaje", "")

    # 1. Limpiar y tokenizar el mensaje (usa tu función clean_text)
    mensaje_limpio = clean_text(mensaje)
    seq = inp_tokenizer.texts_to_sequences([mensaje_limpio])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=config["max_length_inp"], padding='post')

    # 2. Generar respuesta
    hidden = encoder.initialize_hidden_state()
    enc_output, enc_hidden = encoder(padded, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_tokenizer.word_index['<sos>']], 0)

    resultado = ""
    for _ in range(config["max_length_targ"]):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        predicted_id = tf.argmax(predictions[0]).numpy()
        palabra = targ_tokenizer.index_word.get(predicted_id, '')
        if palabra == '<eos>':
            break
        resultado += palabra + " "
        dec_input = tf.expand_dims([predicted_id], 0)

    return jsonify({
        "respuesta": resultado.strip()  # Respuesta en inglés (o idioma original)
    })


@app.route("/api/emocion", methods=["POST"])
def detectar_emocion():
    data = request.json
    texto = data.get("mensaje", "")
    X = vectorizer.transform([texto])
    emocion = modelo_emociones.predict(X)[0]
    return jsonify({"emocion": emocion})

@app.route("/api/traducir", methods=["POST"])
def traducir():
    data = request.json
    texto = data.get("texto", "")
    idioma = data.get("idioma", "es")
    traduccion = GoogleTranslator(source="auto", target=idioma).translate(texto)
    return jsonify({"traduccion": traduccion})

@app.route("/api/botchat", methods=["POST"])
def botchat():
    try:
        data = request.get_json()
        mensaje = data.get("mensaje", "").strip()
        emocion = data.get("emocion", "neutral")

        if not mensaje:
            return jsonify({"error": "Mensaje vacío"}), 400

        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': GEMINI_API_KEY
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"""Responde como un amigo humano del siglo XXI. Usa lenguaje natural, cálido y directo. Sé breve (máximo 30 palabras). 
                            No valides emociones ni uses expresiones como “eso duele” o “te entiendo”.
                            No uses emojis ni jerga informal. Comienza con minúscula, evita el uso excesivo de comas y puntos,
                            Dame solo una opción, en base al mensaje, no uses ninguna expresión si es posible no expreses emociones solo responde

                              Sé claro y sencillo. Mensaje: "{mensaje}"""
                        }
                    ]
                }
            ]
        }

        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            error_msg = response.json().get('error', {}).get('message', 'Error desconocido')
            raise ValueError(f"Error de API: {error_msg}")

        respuesta = response.json()['candidates'][0]['content']['parts'][0]['text']
        
        return jsonify({
            "respuesta": respuesta,
            "status": "success"
        })

    except Exception as e:
        print(f"Error en backend: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500
    
# Iniciar servidor
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto que Render define
    app.run(host="0.0.0.0", port=port, debug=True)

