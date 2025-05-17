# Constantes de configuración
DISTANCIA_HOMBROS_REF = 0.4  # En metros
HISTORIAL_FRAMES = 10  # Número de frames para promediar resultados

# Configuración de MediaPipe
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Umbrales de visibilidad
VISIBILITY_THRESHOLDS = {
    'shoulders': 0.5,
    'nose': 0.5,
    'heels': 0.3
}

# Parámetros para clasificación de tipo de cuerpo
BODY_TYPE_THRESHOLDS = {
    'delgado': 0.22,
    'promedio': 0.26
}

BODY_TYPE_IMC = {
    'delgado': 19,
    'promedio': 22,
    'robusto': 27
}