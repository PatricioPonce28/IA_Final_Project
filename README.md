AMICUSIA - Sistema de Chatbot Empático
<img width="886" height="409" alt="image" src="https://github.com/user-attachments/assets/515082c8-74bb-48b4-8a18-88fdd80acbbe" />
<img width="886" height="403" alt="image" src="https://github.com/user-attachments/assets/1f3081c0-4043-4e7b-8454-f7b6b7ef2696" />
<img width="886" height="405" alt="image" src="https://github.com/user-attachments/assets/84f6cc73-60b2-4a98-a52c-f6ca6c62e3d4" />
<img width="886" height="460" alt="image" src="https://github.com/user-attachments/assets/d971f288-8187-457b-b8ab-7fb147449e42" />

Descripción del Proyecto
AMICUSIA es un sistema de chatbot inteligente diseñado para brindar respuestas empáticas basadas en el análisis emocional de los mensajes de los usuarios. Utiliza técnicas avanzadas de procesamiento de lenguaje natural (NLP) para detectar emociones específicas y generar respuestas contextualmente apropiadas que demuestran comprensión y empatía genuina.
Características Principales

Análisis Emocional Avanzado: Detecta emociones específicas, no solo polaridad básica
Respuestas Empáticas: Genera mensajes contextualmente apropiados según el estado emocional
API REST Completa: Endpoints para análisis, generación y métricas
Interfaz Intuitiva: Frontend amigable para interacción usuario-sistema
Dashboard de Métricas: Monitoreo en tiempo real del rendimiento del sistema
Procesamiento en Tiempo Real: Análisis y respuesta instantáneos

Arquitectura del Sistema
Capa de Datos

Dataset Principal: Empathetic Dialogues (Facebook AI) - 25,000 conversaciones
Preprocesamiento: Limpieza, tokenización y normalización de texto

Capa de Modelo

Clasificación Emocional: Modelo de aprendizaje supervisado con NLP
Análisis de Sentimientos: Determinación de polaridad emocional
Generación de Respuestas: Sistema de creación de mensajes empáticos

Capa de API

Framework: Python con Flask/FastAPI
Endpoints: Análisis, generación, métricas y configuración

Capa de Presentación

Frontend: Interfaz gráfica intuitiva
Dashboard: Visualización de métricas y estadísticas

<img width="914" height="565" alt="image" src="https://github.com/user-attachments/assets/36f49304-28df-469c-a3c4-e8d7c441848e" />

Instalación
1. Clonar el Repositorio
bashgit clone https://github.com/usuario/amicusia.git
cd amicusia

3. Crear Entorno Virtual
bashpython -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

5. Instalar Dependencias
bashpip install -r requirements.txt

7. Configurar Variables de Entorno
bashcp .env.example .env
# Editar .env con tus configuraciones

5. Preparar Datos
bashpython scripts/download_dataset.py
python scripts/preprocess_data.py
Uso
Iniciar el Servidor API
bashpython app.py

La API estará disponible en https://ia-final-project.onrender.com
Ejecutar la Interfaz Web

Link de la presentacion. - https://www.canva.com/design/DAGubaYCKw4/fnSk9bMKb0ewrnEd_JIIjw/edit?utm_content=DAGubaYCKw4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 
<img width="340" height="492" alt="image" src="https://github.com/user-attachments/assets/48ea8ca8-5879-4339-8a08-811b708814ad" />

