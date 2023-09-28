import cv2
import numpy as np

cap = cv2.VideoCapture(r".\detectarCores\video.mp4")
target_color = [0, 182, 252]    #bgr
#rgba(252,182,0,255)

def findRangeHSV(bgr, tresh=40):
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]  # Converte cor em RGB para HSV

    min = np.array([hsv[0] - tresh, hsv[1] - tresh, hsv[2] - tresh])
    max = np.array([hsv[0] + tresh, hsv[1] + tresh, hsv[2] + tresh])

    return min, max

def createMask(img, min, max):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, min, max)

    kernel = np.ones((10, 10), np.uint8)
    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return close


def init():
    min, max = findRangeHSV(target_color, tresh=20)

    while cap.isOpened():
        ok, frame = cap.read()

        if not ok:
            print("Erro no vídeo")
            break

        frame = cv2.resize(frame, (720, 400), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)


        mask = createMask(frame, min, max)
        cv2.imshow("mask", mask)

        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "CONE", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 200), thickness=1)

        cv2.imshow("frame", frame)

        if cv2.waitKey(20) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    init()