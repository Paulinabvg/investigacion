import cv2

def draw_text_with_background(image, text, org, font, scale, color, 
                             thickness=1, bg_color=(0, 0, 0), alpha=0.5):
    """Dibuja texto con fondo semitransparente."""
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - text_h - 10), (x + text_w + 10, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.putText(image, text, (x + 5, y - 5), font, scale, color, thickness, cv2.LINE_AA)
    return image

def clasificar_altura(altura):
    """Clasifica la altura en categorías."""
    if altura > 1.80:
        return "Persona alta", (0, 255, 0)
    elif altura < 1.60:
        return "Persona baja", (0, 0, 255)
    return "Estatura media", (0, 255, 255)