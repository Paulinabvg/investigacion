import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Ajustes de calibración (debes modificar estos valores según tu cámara)
ALTURA_CAMARA_METROS = 1.5  # Altura de la cámara desde el suelo
PERSONA_REF_ALTURA = 1.75  # Altura de una persona de referencia (para calibrar)
PERSONA_REF_PX = 500       # Píxeles que ocupa esa persona en la imagen (ajustar manualmente)

# Calcular relación píxeles/metros (usando calibración)
px_por_metro = PERSONA_REF_PX / PERSONA_REF_ALTURA

cap = cv2.VideoCapture(2)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede acceder a la cámara")
        break
    cv2.imshow("Test Cam", frame)
    if cv2.waitKey(1) == 27:
        break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark

        # Puntos clave: nariz y talón izquierdo (o derecho)
        y_nose = landmarks[mp_pose.PoseLandmark.NOSE].y * h
        y_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * h

        # Distancia en píxeles entre nariz y talón
        altura_px = y_heel - y_nose

        # Estimación de altura (ajustada con calibración)
        altura_metros = altura_px / px_por_metro

        # Mostrar resultados
        cv2.putText(frame, f"Altura estimada: {altura_metros:.2f} m", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)

        if altura_metros > 1.80:
            cv2.putText(frame, "Persona alta", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    cv2.imshow("Altura estimada con MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()