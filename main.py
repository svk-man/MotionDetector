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

    # Определить различия между первым кадром и остальными
    delta_frame = cv2.absdiff(first_frame, gray)
    cv2.imshow('frame-delta', delta_frame)

    # Преобразовать кадр в оттенках серого в черно-белый
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # Расширить светлые области и сузить темные
    thresh_delta = cv2.dilate(delta_frame, None, iterations=0)
    cv2.imshow('frame-dilate', thresh_delta)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
