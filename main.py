import numpy as np
import cv2

first_frame = None

cap = cv2.VideoCapture('video/video#1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))

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
    thresh_delta = cv2.threshold(delta_frame, 35, 255, cv2.THRESH_BINARY)[1]

    # Расширить светлые области и сузить темные
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=2)
    cv2.imshow('frame-dilate', thresh_delta)

    # Поиск контуров
    (cnts, _) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('frame-result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
