import cv2
import mediapipe as mp
import argparse
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

DISTANCIA_HOMBROS_REF = 0.4  # En metros

def calcular_factor_escala(landmarks, frame_shape):
    h, w = frame_shape[:2]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_distance_px = np.sqrt(
        (right_shoulder.x * w - left_shoulder.x * w)**2 +
        (right_shoulder.y * h - left_shoulder.y * h)**2
    )
    return DISTANCIA_HOMBROS_REF / shoulder_distance_px if shoulder_distance_px > 0 else 1.0

def estimar_altura(landmarks, frame_shape, scale_factor):
    h, w = frame_shape[:2]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
    y_heel = max(left_heel.y, right_heel.y) * h
    y_nose = nose.y * h
    altura_px = y_heel - y_nose
    altura_metros = altura_px * scale_factor * 1.15
    return altura_metros, altura_px

def estimar_tipo_cuerpo(landmarks, frame_shape, altura_px):
    h, w = frame_shape[:2]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_distance_px = np.sqrt(
        (right_shoulder.x * w - left_shoulder.x * w) ** 2 +
        (right_shoulder.y * h - left_shoulder.y * h) ** 2
    )
    proporcion = shoulder_distance_px / altura_px
    if proporcion < 0.22:
        return "Delgado", 19
    elif proporcion < 0.26:
        return "Promedio", 22
    else:
        return "Robusto/Obeso", 27

def draw_text_with_background(image, text, org, font, scale, color, thickness=1, bg_color=(0, 0, 0), alpha=0.5):
    """Dibuja texto con fondo semitransparente."""
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - text_h - 10), (x + text_w + 10, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.putText(image, text, (x + 5, y - 5), font, scale, color, thickness, cv2.LINE_AA)

def procesar_imagen(ruta_imagen):
    frame = cv2.imread(ruta_imagen)
    if frame is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen}")
        return

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        scale_factor = calcular_factor_escala(landmarks, frame.shape)
        altura_metros, altura_px = estimar_altura(landmarks, frame.shape, scale_factor)
        tipo_cuerpo, imc_aprox = estimar_tipo_cuerpo(landmarks, frame.shape, altura_px)
        peso_estimado = imc_aprox * (altura_metros ** 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bg_color = (0, 0, 0)
        alpha = 0.5

        draw_text_with_background(frame, f"Altura estimada: {altura_metros:.2f} m", (30, 60),
                                  font, 1, (255, 215, 0), 2, bg_color, alpha)

        if altura_metros > 1.80:
            clasificacion = "Persona alta"
            color = (0, 255, 0)
        elif altura_metros < 1.60:
            clasificacion = "Persona baja"
            color = (0, 0, 255)
        else:
            clasificacion = "Estatura media"
            color = (0, 255, 255)

        draw_text_with_background(frame, clasificacion, (30, 110),
                                  font, 1.1, color, 2, bg_color, alpha)

        draw_text_with_background(frame, f"Tipo de cuerpo: {tipo_cuerpo}", (30, 160),
                                  font, 1, (200, 200, 255), 2, bg_color, alpha)

        draw_text_with_background(frame, f"Peso estimado: {peso_estimado:.1f} kg", (30, 210),
                                  font, 1, (180, 180, 180), 2, bg_color, alpha)

        draw_text_with_background(frame, f"Escala: {scale_factor:.5f} m/px", (30, 260),
                                  font, 0.7, (200, 200, 200), 1, bg_color, alpha)

    cv2.imshow("Analisis corporal con MediaPipe", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimar altura y tipo corporal desde imagen')
    parser.add_argument('imagen', type=str, help='Ruta de la imagen a procesar')
    args = parser.parse_args()
    procesar_imagen(args.imagen)
