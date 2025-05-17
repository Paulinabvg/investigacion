import argparse
import cv2
from investigacion.estimacionPose import PoseEstimator
from investigacion.bodyCalculator import BodyCalculator
from investigacion.utils import draw_text_with_background, clasificar_altura
from investigacion.config import HISTORIAL_FRAMES

def procesar_video(fuente_video):
    pose_estimator = PoseEstimator()
    body_calculator = BodyCalculator(buffer_size=HISTORIAL_FRAMES)
    
    cap = cv2.VideoCapture(fuente_video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la fuente de video {fuente_video}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = pose_estimator.process_frame(frame)
        frame = pose_estimator.draw_landmarks(frame, landmarks)
        
        if landmarks:
            scale_factor = body_calculator.calcular_factor_escala(landmarks.landmark, frame.shape)
            
            if scale_factor is not None:
                altura, altura_px = body_calculator.estimar_altura(landmarks.landmark, frame.shape, scale_factor)
                
                if altura is not None and altura_px is not None:
                    tipo_cuerpo, imc_aprox = body_calculator.estimar_tipo_cuerpo(
                        landmarks.landmark, frame.shape, altura_px)
                    
                    if tipo_cuerpo is not None and imc_aprox is not None:
                        peso_estimado = imc_aprox * (altura ** 2)
                        body_calculator.actualizar_buffers(altura, (tipo_cuerpo, imc_aprox), peso_estimado)
        
        # Mostrar resultados promediados
        altura_prom, tipo_prom, peso_prom = body_calculator.obtener_resultados_promediados()
        
        if altura_prom is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bg_color = (0, 0, 0)
            alpha = 0.5
            
            frame = draw_text_with_background(frame, f"Altura estimada: {altura_prom:.2f} m", (30, 60),
                                            font, 1, (255, 215, 0), 2, bg_color, alpha)
            
            clasif, color = clasificar_altura(altura_prom)
            frame = draw_text_with_background(frame, clasif, (30, 110),
                                            font, 1.1, color, 2, bg_color, alpha)
            
            if tipo_prom is not None:
                frame = draw_text_with_background(frame, f"Tipo de cuerpo: {tipo_prom}", (30, 160),
                                                font, 1, (200, 200, 255), 2, bg_color, alpha)
            
            if peso_prom is not None:
                frame = draw_text_with_background(frame, f"Peso estimado: {peso_prom:.1f} kg", (30, 210),
                                                font, 1, (180, 180, 180), 2, bg_color, alpha)
            
            frame = draw_text_with_background(frame, 
                                            f"Frames promediados: {len(body_calculator.alturas)}/{HISTORIAL_FRAMES}",
                                            (30, 260), font, 0.7, (200, 200, 200), 1, bg_color, alpha)
        
        cv2.imshow("Analisis corporal en tiempo real", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimar altura y tipo corporal desde video/cámara')
    parser.add_argument('--video', type=str, help='Ruta del video a procesar (opcional)')
    args = parser.parse_args()
    
    fuente = 0 if args.video is None else args.video  # 0 para cámara por defecto
    procesar_video(fuente)