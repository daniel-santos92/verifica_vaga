import cv2
import numpy as np

# Carregar vídeo
video = cv2.VideoCapture('/home/danielsantos/Documentos/Computação Grafica/Estacionamento.mp4')

# Lista manual de vagas (x, y, largura, altura)
vagas = [
    (10, 230, 275, 450),   # Vaga 1: Da borda até ANTES da 1ª linha azul
    (295, 230, 210, 450),  # Vaga 2: Entre a 1ª e a 2ª linha azul
    (515, 230, 260, 450),  # Vaga 3: Entre a 2ª e a 3ª linha azul
    (785, 230, 240, 450)   # Vaga 4: DEPOIS da 3ª linha azul até a borda
]

# Função para checar se uma vaga está livre com base na área de pixels brancos
def checar_vaga(img_proc, x, y, w, h):
    recorte = img_proc[y:y+h, x:x+w]
    total_pixels_brancos = cv2.countNonZero(recorte)
    area = w * h
    return total_pixels_brancos < area * 0.025  # Threshold relativo à área

while True:
    sucesso, frame = video.read()
    if not sucesso:
        break

    # Pré-processamento
    img_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_cinza, (5, 5), 1)

    # Threshold adaptativo para destacar carros
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 16
    )
    img_median = cv2.medianBlur(img_thresh, 5)

    # Operações morfológicas para limpar a imagem
    kernel = np.ones((3, 3), np.uint8)
    img_close = cv2.morphologyEx(img_median, cv2.MORPH_CLOSE, kernel)
    img_dilate = cv2.dilate(img_close, kernel, iterations=2)

    # Borda adicional com Canny para reforçar contornos
    img_canny = cv2.Canny(img_blur, 100, 200)
    img_combined = cv2.bitwise_or(img_dilate, img_canny)

    vagas_livres = 0
    vagas_ocupadas = 0

    # Verificar todas as vagas
    for x, y, w, h in vagas:
        livre = checar_vaga(img_combined, x, y, w, h)

        if livre:
            vagas_livres += 1
            cor = (0, 255, 0)
            texto = "Livre"
        else:
            vagas_ocupadas += 1
            cor = (0, 0, 255)
            texto = "Ocupada"

        # Desenhar vaga e status
        cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(frame, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)

    # Exibir contador de vagas
    texto_contador = f"Livres: {vagas_livres} | Ocupadas: {vagas_ocupadas}"
    cv2.rectangle(frame, (10, 10), (320, 45), (0, 0, 0), -1)
    cv2.putText(frame, texto_contador, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostrar resultado
    cv2.imshow("Detecção de Vagas", frame)
    # cv2.imshow("Processado", img_combined)  # Útil para depurar

    if cv2.waitKey(30) & 0xFF == 27:  # Tecla ESC para sair
        break

video.release()
cv2.destroyAllWindows()