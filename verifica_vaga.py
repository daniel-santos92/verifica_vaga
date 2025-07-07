import cv2
import numpy as np

# Caminhos dos arquivos do modelo e do vídeo
video_path = "Estacionamento.mp4"
prototxt_path = "MobileNetSSD_deploy.prototxt"
model_path = "MobileNetSSD_deploy.caffemodel"

# Lista de classes reconhecidas pelo modelo MobileNetSSD
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Coordenadas das vagas (retângulos): (x1, y1, x2, y2)
vagas = [
    (1, 130, 180, 463),
    (230, 80, 430, 470),   # vaga 2 ampliada
    (460, 90, 640, 463),
    (700, 100, 845, 463),
]

# Função para detectar a cor predominante no centro da imagem da vaga
def detectar_cor_predominante(roi):
    h, w = roi.shape[:2]
    start_x = w // 3
    start_y = h // 3
    end_x = start_x + w // 3
    end_y = start_y + h // 3
    centro = roi[start_y:end_y, start_x:end_x]

    pixels = centro.reshape(-1, 3).astype(np.float32)
    # Critérios de parada para o algoritmo KMeans:
        # - TERM_CRITERIA_EPS: para quando o movimento do centro for pequeno (precisão)
        # - TERM_CRITERIA_MAX_ITER: para quando atingir número máximo de iterações (10)
        # - O terceiro parâmetro (1.0) é a precisão esperada
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Executa o algoritmo KMeans:
    # - pixels: dados de entrada
    # - 2: número de clusters (cores predominantes)
    # - None: sem labels iniciais
    # - criteria: critério de parada (EPS + MAX_ITER)
    # - 10: número de tentativas com inicializações diferentes
    # - KMEANS_RANDOM_CENTERS: inicializa os centros de forma aleatória
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    cor = centers[np.argmax(counts)].astype(int)

    hsv = cv2.cvtColor(np.uint8([[cor]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    if v < 50:
        return "Preto"
    if s < 40:
        return "Branco" if v > 180 else "Cinza"
    if h < 10 or h > 160:
        return "Vermelho"
    if h <= 25:
        return "Laranja"
    if h <= 35:
        return "Amarelo"
    if h <= 85:
        return "Verde"
    if h <= 130:
        return "Azul"
    if h <= 160:
        return "Roxo"
    return "Desconhecida"

# Carrega o modelo de detecção treinado
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Abre o vídeo
cap = cv2.VideoCapture(video_path)

# Usa o primeiro frame como fundo estático para comparação de mudança
ret, primeiro_frame = cap.read()
fundo_estatico = primeiro_frame.copy() if ret else None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vagas_ocupadas = 0

    for i, (x1, y1, x2, y2) in enumerate(vagas):
        roi = frame[y1:y2, x1:x2]
        w, h = x2 - x1, y2 - y1

        # Prepara o recorte da vaga para detecção com MobileNet
        blob = cv2.dnn.blobFromImage(roi, 0.007843, (w, h), 127.5)
        net.setInput(blob)
        detections = net.forward()

        objeto_detectado = "Livre"
        for j in range(detections.shape[2]):
            conf = detections[0, 0, j, 2]
            if conf > 0.25:  # menor confiança para mais sensibilidade
                idx = int(detections[0, 0, j, 1])
                if CLASSES[idx] in ["car", "person", "bicycle", "motorbike"]:
                    objeto_detectado = CLASSES[idx]
                    break

        # Se nada foi detectado, faz subtração de fundo para detectar mudança visual
        if objeto_detectado == "Livre" and fundo_estatico is not None:
            fundo_roi = fundo_estatico[y1:y2, x1:x2]
            diff = cv2.absdiff(roi, fundo_roi)
            cinza = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(cinza, 30, 255, cv2.THRESH_BINARY)
            movimento = cv2.countNonZero(mask)
            if movimento > 1000:
                objeto_detectado = "Oculto"

        # Se ocupado, conta e detecta cor predominante
        if objeto_detectado != "Livre":
            vagas_ocupadas += 1
            cor_predominante = detectar_cor_predominante(roi)
        else:
            cor_predominante = ""

        # Desenha o retângulo da vaga
        status = "Ocupada" if objeto_detectado != "Livre" else "Livre"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Vaga {i+1}: {status}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Se ocupado, escreve tipo e cor do objeto detectado
        if objeto_detectado != "Livre":
            cv2.putText(frame, f"Tipo: {objeto_detectado}", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Cor: {cor_predominante}", (x1 + 5, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Mostra o total de vagas ocupadas
    cv2.putText(frame, f"Vagas Ocupadas: {vagas_ocupadas}/4", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Exibe o resultado
    cv2.imshow("Monitoramento de Vagas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:  # sair com Q ou ESC
        break

cap.release()
cv2.destroyAllWindows()
