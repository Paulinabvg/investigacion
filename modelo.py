import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Ajusta segun la altura real a la que este colocada tu camara
ALTURA_CAMARA_METROS = 1.5  # en metros

# Asumimos que la camara esta perpendicular al suelo
cap = cv2.VideoCapture(0)

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

        # Puntos clave: nariz y talón izquierdo
        y_nose = landmarks[mp_pose.PoseLandmark.NOSE].y * h
        y_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * h

        # Distancia en píxeles entre la cabeza y los pies
        altura_px = y_heel - y_nose

        # Estimacion proporcional: 1 metro ≈ X píxeles
        # Esto requiere una calibracion. Puedes medir a alguien de altura conocida (ej. 1.75 m) y ver cuantos pixeles mide en la imagen.
        # Aquí ponemos un valor aproximado calibrado empiricamente:
        px_por_metro = 520  # Ejemplo: en tu camara, 1 metro = 520 pixeles (ajusta esto)

        altura_metros = altura_px / px_por_metro

        cv2.putText(frame, f"Altura estimada: {altura_metros:.2f} m", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)

        if altura_metros > 1.80:
            cv2.putText(frame, "Persona alta", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    cv2.imshow("Altura real estimada", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
