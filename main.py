import numpy as np
import cv2

first_frame = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)

    # Размытие фильтром Гаусса
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    cv2.imshow('frame-blur', gray)

    # Сохранить первый кадр видео
    if first_frame is None:
        first_frame = gray
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
