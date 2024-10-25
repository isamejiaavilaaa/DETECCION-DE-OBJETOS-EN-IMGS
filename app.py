import cv2
import streamlit as st
import numpy as np
import pandas as pd
from ultralytics import YOLO  # Usar ultralytics en lugar de yolov5

# Cargar el modelo YOLOv5s
model = YOLO('yolov5s.pt')  # Asegúrate de tener el archivo yolov5s.pt en tu directorio

# Ajustar los parámetros del modelo
model.conf = 0.25  # Umbral de confianza de NMS
model.iou = 0.45  # Umbral de IoU de NMS
model.agnostic = False  # NMS no clasifica por clases
model.multi_label = False  # NMS no usa múltiples etiquetas por caja
model.max_det = 1000  # Número máximo de detecciones por imagen

# Configurar la interfaz en Streamlit
st.title("Detección de Objetos en Imágenes")

# Configurar el menú lateral con los sliders para IoU y confianza
with st.sidebar:
    st.subheader('Parámetros de Configuración')
    model.iou = st.slider('Seleccione el IoU', 0.0, 1.0, value=0.45)
    st.write('IOU:', model.iou)

with st.sidebar:
    model.conf = st.slider('Seleccione el Confidence', 0.0, 1.0, value=0.25)
    st.write('Conf:', model.conf)

# Tomar una imagen con la cámara
picture = st.camera_input("Capturar foto", label_visibility='visible')

