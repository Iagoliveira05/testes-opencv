import cv2
import sys
from random import randint

def redim(img, largura): #função para redimensionar uma imagem
    alt = int(img.shape[0]/img.shape[1]*largura)
    img = cv2.resize(img, (largura, alt), interpolation =cv2.INTER_AREA)
    return img



cap = cv2.VideoCapture(r".\detectarVideo\video2.mp4")

ok, frame = cap.read()
if not ok:
    print("Erro ao ler o vídeo")
    sys.exit(1)

bboxes = []
colors = []

# frame = cv2.resize(frame, (800, 600))
frame = redim(frame, 320)
while True:
    bbox = cv2.selectROI("Tracker", frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Pressione Q para sair e qualquer outra para continuar")
    key = cv2.waitKey(0) & 0XFF
    if key == 113:
        break

tracker = cv2.legacy.TrackerCSRT_create()
multitracker = cv2.legacy.MultiTracker_create()

for bbox in bboxes:
    multitracker.add(tracker, frame, bbox)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    
    frame = redim(frame, 320)

    # frame = cv2.resize(frame, (800, 600))
    ok, boxes = multitracker.update(frame)

    for i, newbox in enumerate(boxes):
        (x,y,w,h) = [int(v) for v in newbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), colors[i], 2, 3)


    cv2.imshow("Multitracker", frame)

    if cv2.waitKey(1) & 0XFF == 27:
        break
