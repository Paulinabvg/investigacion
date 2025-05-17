import numpy as np
import mediapipe as mp
from collections import deque

from .config import (
    DISTANCIA_HOMBROS_REF, 
    VISIBILITY_THRESHOLDS,
    BODY_TYPE_THRESHOLDS,
    BODY_TYPE_IMC
)

class BodyCalculator:
    def __init__(self, buffer_size=10):
        self.alturas = deque(maxlen=buffer_size)
        self.tipos_cuerpo = deque(maxlen=buffer_size)
        self.pesos = deque(maxlen=buffer_size)
        
    def calcular_factor_escala(self, landmarks, frame_shape):
        """Calcula el factor de escala basado en la distancia entre hombros."""
        h, w = frame_shape[:2]
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        
        if (left_shoulder.visibility < VISIBILITY_THRESHOLDS['shoulders'] or 
            right_shoulder.visibility < VISIBILITY_THRESHOLDS['shoulders']):
            return None
        
        shoulder_distance_px = np.sqrt(
            (right_shoulder.x * w - left_shoulder.x * w)**2 +
            (right_shoulder.y * h - left_shoulder.y * h)**2
        )
        return DISTANCIA_HOMBROS_REF / shoulder_distance_px if shoulder_distance_px > 0 else None

    def estimar_altura(self, landmarks, frame_shape, scale_factor):
        """Estima la altura basada en los landmarks."""
        h, w = frame_shape[:2]
        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]
        left_heel = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HEEL]
        right_heel = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HEEL]
        
        if (nose.visibility < VISIBILITY_THRESHOLDS['nose'] or 
            left_heel.visibility < VISIBILITY_THRESHOLDS['heels'] or 
            right_heel.visibility < VISIBILITY_THRESHOLDS['heels']):
            return None, None
        
        y_heel = max(left_heel.y, right_heel.y) * h
        y_nose = nose.y * h
        altura_px = y_heel - y_nose
        altura_metros = altura_px * scale_factor * 1.15  # Factor de corrección
        return altura_metros, altura_px

    def estimar_tipo_cuerpo(self, landmarks, frame_shape, altura_px):
        """Estima el tipo de cuerpo basado en proporciones."""
        h, w = frame_shape[:2]
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        
        if (left_shoulder.visibility < VISIBILITY_THRESHOLDS['shoulders'] or 
            right_shoulder.visibility < VISIBILITY_THRESHOLDS['shoulders'] or
            altura_px <= 0):
            return None, None
        
        shoulder_distance_px = np.sqrt(
            (right_shoulder.x * w - left_shoulder.x * w)**2 +
            (right_shoulder.y * h - left_shoulder.y * h)**2
        )
        
        proporcion = shoulder_distance_px / altura_px
        
        if proporcion < BODY_TYPE_THRESHOLDS['delgado']:
            return "Delgado", BODY_TYPE_IMC['delgado']
        elif proporcion < BODY_TYPE_THRESHOLDS['promedio']:
            return "Promedio", BODY_TYPE_IMC['promedio']
        else:
            return "Robusto/Obeso", BODY_TYPE_IMC['robusto']

    def actualizar_buffers(self, altura, tipo_cuerpo, peso):
        """Actualiza los buffers con los nuevos valores."""
        if altura is not None:
            self.alturas.append(altura)
        if tipo_cuerpo is not None:
            self.tipos_cuerpo.append(tipo_cuerpo)
        if peso is not None:
            self.pesos.append(peso)

    def obtener_resultados_promediados(self):
        """Devuelve los resultados promediados de los buffers."""
        if not self.alturas:
            return None, None, None
        
        altura_prom = sum(self.alturas) / len(self.alturas)
        
        # Calcular moda para tipo de cuerpo
        tipos = [t[0] for t in self.tipos_cuerpo]
        tipo_prom = max(set(tipos), key=tipos.count) if tipos else None
        
        peso_prom = sum(self.pesos) / len(self.pesos) if self.pesos else None
        
        return altura_prom, tipo_prom, peso_prom