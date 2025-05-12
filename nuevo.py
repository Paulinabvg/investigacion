import cv2
import mediapipe as mp
import argparse
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

DISTANCIA_HOMBROS_REF = 0.4  # Distancia aproximada entre hombros en metros (para persona de 1.75m)
ALTURA_PROMEDIO = 1.75

def calcular_factor_escala(landmarks, frame_shape):
    """Calcula el factor de escala basado en la distancia entre hombros."""
    h, w = frame_shape[:2]
    
    # Obtener coordenadas de los hombros
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    
    # Calcular distancia entre hombros en píxeles
    shoulder_distance_px = np.sqrt(
        (right_shoulder.x * w - left_shoulder.x * w)**2 + 
        (right_shoulder.y * h - left_shoulder.y * h)**2
    )
    
    # Calcular factor de escala (metros por píxel)
    if shoulder_distance_px > 0:
        scale_factor = DISTANCIA_HOMBROS_REF / shoulder_distance_px
    else:
        scale_factor = 1.0  # Valor por defecto si no se detectan hombros
    
    return scale_factor

def estimar_altura(landmarks, frame_shape, scale_factor):
    """Estima la altura usando el factor de escala calculado."""
    h, w = frame_shape[:2]
    
    # Puntos clave para altura: usar NOSE (nariz) en lugar de HEAD
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
    
    # Usar el talón más bajo (en caso de que una pierna esté levantada)
    y_heel = max(left_heel.y, right_heel.y) * h
    y_nose = nose.y * h
    
    altura_px = y_heel - y_nose
    
    # Convertir a metros usando el factor de escala
    altura_metros = altura_px * scale_factor
    
    # Ajuste basado en proporciones corporales (la nariz está por debajo de la parte superior de la cabeza)
    # Agregamos aproximadamente 15cm para compensar
    altura_metros *= 1.15
    
    return altura_metros

def procesar_imagen(ruta_imagen):
    # Leer la imagen
    frame = cv2.imread(ruta_imagen)
    if frame is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return
    
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar con MediaPipe
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Dibujar landmarks
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Calcular factor de escala basado en distancia entre hombros
        scale_factor = calcular_factor_escala(landmarks, frame.shape)
        
        # Estimar altura
        altura_metros = estimar_altura(landmarks, frame.shape, scale_factor)
        
        # Mostrar resultados
        cv2.putText(frame, f"Altura estimada: {altura_metros:.2f} m", (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)
        
        # Clasificación de estatura
        if altura_metros > 1.80:
            clasificacion = "Persona alta"
            color = (0, 255, 0)
        elif altura_metros < 1.60:
            clasificacion = "Persona baja"
            color = (0, 0, 255)
        else:
            clasificacion = "Estatura media"
            color = (0, 255, 255)
            
        cv2.putText(frame, clasificacion, (30, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        
        # Mostrar también la distancia entre hombros como referencia
        shoulder_px = 1 / scale_factor * DISTANCIA_HOMBROS_REF if scale_factor != 0 else 0
        cv2.putText(frame, f"Escala: {scale_factor:.5f} m/px", (30, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # Mostrar la imagen procesada
    cv2.imshow("Altura estimada con MediaPipe", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimar altura de persona en una imagen')
    parser.add_argument('imagen', type=str, help='Ruta de la imagen a procesar')
    args = parser.parse_args()
    
    procesar_imagen(args.imagen)