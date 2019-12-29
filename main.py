import numpy as np
import cv2
import xml.etree.ElementTree as ET


def save_to_file(filename, content):
    fh = None
    try:
        fh = open(filename, "w", encoding="utf-8")
        fh.write(content)
    except Exception as e:
        print("Ошибка при работе с файлом:", e)
    finally:
        if fh:
            fh.close()


def create_xml(filename):
    annotation = ET.Element('annotation')

    ET.SubElement(annotation, "folder").text = "xgRect"
    ET.SubElement(annotation, "filename").text = "405.png"
    ET.SubElement(annotation, "path").text = "/home/coldmoon/Database/logo_qiaowei/xgRect/405.png"

    source = ET.Element("source")
    ET.SubElement(source, "database").text = "Unknown"
    annotation.append(source)

    size = ET.Element("size")
    ET.SubElement(size, "width").text = "850"
    ET.SubElement(size, "height").text = "480"
    ET.SubElement(size, "depth").text = "3"
    annotation.append(size)

    ET.SubElement(annotation, "segmented").text = "0"

    object = ET.Element("object")
    ET.SubElement(object, "name").text = "widemelon"
    ET.SubElement(object, "pose").text = "Unspecified"
    ET.SubElement(object, "truncated").text = "0"
    ET.SubElement(object, "difficult").text = "0"
    bndbox = ET.Element("bndbox")
    ET.SubElement(bndbox, "xmin").text = "657"
    ET.SubElement(bndbox, "ymin").text = "19"
    ET.SubElement(bndbox, "xmax").text = "827"
    ET.SubElement(bndbox, "ymax").text = "74"
    object.append(bndbox)
    annotation.append(object)

    tree = ET.ElementTree(annotation)
    tree.write(filename)


first_frame = None

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('video/video#1.mp4')

while True:
    # while cap.isOpened():
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (720, 480))

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

    # Найти максимальное значение площади и выделить объект прямоугольником
    if cnts:
        area_max = 0
        index = 0
        index_max_area = 0
        for contour in cnts:
            area = cv2.contourArea(contour)
            if area > area_max:
                area_max = area
                index_max_area = index
            index = index + 1

        cnt = cnts[index_max_area]
        x, y, w, h = cv2.boundingRect(cnt)
        image = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame-result', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

create_xml("test.xml")

