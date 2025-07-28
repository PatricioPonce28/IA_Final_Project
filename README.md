Link Canva. - https://www.canva.com/design/DAGubaYCKw4/fnSk9bMKb0ewrnEd_JIIjw/edit?utm_content=DAGubaYCKw4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 
📖 Descripción
AmicusIA es una inteligencia artificial diseñada como amigo virtual que proporciona acompañamiento emocional a través del reconocimiento de sentimientos en mensajes de texto. Utilizando técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático, el sistema identifica el estado emocional del usuario y genera respuestas empáticas personalizadas.
✨ Características Principales

🎯 Análisis Emocional en Tiempo Real: Clasificación de emociones con 85.3% de precisión
💬 Conversación Empática: Respuestas contextualizadas según el estado emocional detectado
🧠 Arquitectura Dual: Combina clasificación emocional y generación conversacional
🌐 API REST: Interfaz para integración con aplicaciones web y móviles
📊 Métricas en Tiempo Real: Visualización del rendimiento del modelo

🏗️ Arquitectura del Sistema
Componentes:

Modelo de Clasificación Emocional: Regresión Logística con vectorización TF-IDF
Modelo Conversacional: Arquitectura Encoder-Decoder con LSTM
API Backend: Exposición de endpoints para análisis y conversación
Frontend: Interfaz de usuario intuitiva para interacción

📊 Rendimiento del Modelo
ModeloAlgoritmoPrecisión/PérdidaDatasetClasificación EmocionalRegresión Logística85.3%Emotions69k.csv (69k+ registros)ConversacionalSeq2Seq LSTM0.0303 (pérdida final)dialogs.txt
Progreso de Entrenamiento:
Época  4: Pérdida 1.5570
Época  8: Pérdida 1.3285
Época 12: Pérdida 1.1333
...
Época 40: Pérdida 0.0303 ✅
🚀 Instalación y Configuración
Prerrequisitos

Python 3.8 o superior
pip (gestor de paquetes de Python)

1. Clonar el Repositorio
bashgit clone https://github.com/Odaliz2105/ProyectoIA.git
cd ProyectoIA
2. Crear Entorno Virtual (Recomendado)
bashpython -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
3. Instalar Dependencias
bashpip install -r requirements.txt
Dependencias Principales:
txtpandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
transformers>=4.0.0
flask>=2.0.0
matplotlib>=3.5.0
numpy>=1.21.0
joblib>=1.1.0
deep-translator>=1.8.0
tqdm>=4.62.0
4. Preparar Datasets
Asegúrate de tener los siguientes archivos en el directorio del proyecto:

Emotions69k.csv: Dataset de emociones
dialogs.txt: Dataset conversacional

🎮 Uso del Sistema
Entrenamiento de Modelos
1. Modelo de Clasificación Emocional
pythonimport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Cargar dataset
df = pd.read_csv("Emotions69k.csv", sep=";", encoding="utf-8", 
                 quotechar='"', on_bad_lines="skip")

# Entrenar modelo
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Situation"])
y = df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo_emocion = LogisticRegression(max_iter=1000)
modelo_emocion.fit(X_train, y_train)
2. Modelo Conversacional
python# Entrenar modelo Seq2Seq (ver notebooks para implementación completa)
python train_conversational_model.py
Ejecutar la API
bashpython app.py
La API estará disponible en: http://localhost:5000
Endpoints Disponibles
EndpointMétodoDescripción/analyzePOSTAnálisis de sentimiento de un mensaje/chatPOSTConversación completa con respuesta empática/metricsGETMétricas del modelo y estadísticas/healthGETEstado del sistema
Ejemplo de Uso de la API
pythonimport requests

# Análisis emocional
response = requests.post('http://localhost:5000/analyze', 
                        json={'message': 'Me siento muy triste hoy'})
print(response.json())
# Output: {'emotion': 'sadness', 'confidence': 0.89}

# Conversación
response = requests.post('http://localhost:5000/chat', 
                        json={'message': 'Hola, ¿cómo estás?'})
print(response.json())
# Output: {'response': 'Hola! Estoy bien, gracias por preguntar. ¿Cómo te sientes hoy?', 'emotion': 'neutral'}

🧪 Pruebas
Ejecutar Pruebas Unitarias
bashpython -m pytest tests/
Validar Modelos
bashpython src/validate_models.py
Ejemplo de Prueba Manual
python# Probar clasificación emocional
test_messages = [
    "Me siento muy feliz hoy",
    "Estoy triste y cansado",
    "¡Qué sorpresa tan increíble!",
    "Tengo miedo de lo que pueda pasar"
]

for message in test_messages:
    emotion = modelo_emocion.predict(vectorizer.transform([message]))[0]
    print(f"Mensaje: {message}")
    print(f"Emoción detectada: {emotion}\n")
👥 Equipo de Desarrollo
DesarrolladorRolContribucionesOdaliz BalsecaML EngineerDesarrollo del modelo de IA y análisis de datosPatricio PonceBackend DeveloperImplementación de la API y arquitectura backendAlisson ViracochaFrontend DeveloperDesarrollo de interfaz e integración del sistema
📈 Métricas de Contribución

Total de commits: 45+
Issues resueltos: 12
Pull requests: 15
Líneas de código: 2,500+

🛠️ Tecnologías Utilizadas
Backend

Python 3.8+: Lenguaje principal
TensorFlow/Keras: Modelos de deep learning
Scikit-learn: Machine learning tradicional
Flask/FastAPI: API REST
Pandas: Manipulación de datos

Frontend

HTML5/CSS3: Estructura y estilos
JavaScript: Interactividad
Bootstrap: Framework CSS

Herramientas de Desarrollo

Git/GitHub: Control de versiones
Jupyter Notebooks: Experimentación
Matplotlib: Visualización de datos

🎯 Casos de Uso
1. Apoyo Emocional Personal

Usuarios que buscan compañía virtual
Personas que necesitan una primera escucha empática
Individuos en proceso de autoconocimiento emocional

2. Integración en Aplicaciones

Chatbots de atención al cliente con componente emocional
Aplicaciones de salud mental y bienestar
Plataformas educativas con soporte emocional

3. Investigación y Desarrollo

Estudios sobre análisis de sentimientos
Desarrollo de sistemas conversacionales
Investigación en IA empática
