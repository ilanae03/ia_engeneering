# RM = 92884 = 9+2+8+8+4 = 31 = 3+1 = 4 (video q1A.mp4)
import cv2
import numpy as np

def detectar_formas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    return [(mask_red, (0, 0, 255)), (mask_blue, (255, 0, 0))]

def encontrar_maior_forma(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maior_contorno = None
    maior_area = 0
    
    for contorno in contours:
        area = cv2.contourArea(contorno)
        if area > maior_area:
            maior_area = area
            maior_contorno = contorno
    
    return maior_contorno, maior_area

def verificar_colisao(contorno1, contorno2):
    if contorno1 is None or contorno2 is None:
        return False
    
    x1, y1, w1, h1 = cv2.boundingRect(contorno1)
    x2, y2, w2, h2 = cv2.boundingRect(contorno2)
    
    return not (x1 > x2 + w2 or x2 > x1 + w1 or y1 > y2 + h2 or y2 > y1 + h1)

def verificar_ultrapassagem(contorno1, contorno2):
    if contorno1 is None or contorno2 is None:
        return False
    
    x1, y1, w1, h1 = cv2.boundingRect(contorno1)
    x2, y2, w2, h2 = cv2.boundingRect(contorno2)
    
    return (x1 + w1 < x2 or x2 + w2 < x1) or (y1 + h1 < y2 or y2 + h2 < y1)

cap = cv2.VideoCapture("q1/q1A.mp4")
colisao_ocorreu = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    formas = detectar_formas(frame)
    contornos_e_areas = [encontrar_maior_forma(mask) for mask, _ in formas]
    contornos = [item[0] for item in contornos_e_areas]
    areas = [item[1] for item in contornos_e_areas]
    
    for i, (contorno, cor, area) in enumerate(zip(contornos, [(0, 255, 0), (255, 0, 0)], areas)):
        if contorno is not None:
            x, y, w, h = cv2.boundingRect(contorno)
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
            cv2.putText(frame, f"Massa: {area:.0f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    
    if verificar_colisao(*contornos):
        cv2.putText(frame, "COLISAO DETECTADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        colisao_ocorreu = True
    
    if colisao_ocorreu and verificar_ultrapassagem(*contornos):
        cv2.putText(frame, "ULTRAPASSAGEM DETECTADA", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    
    cv2.imshow("Feed", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
