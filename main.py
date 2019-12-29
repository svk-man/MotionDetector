import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import glob


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


def create_xml(filename, xml_folder, xml_filename, xml_path, xml_size, xml_height, xml_xmin, xml_ymin, xml_xmax, xml_ymax):
    annotation = ET.Element('annotation')

    ET.SubElement(annotation, "folder").text = xml_folder
    ET.SubElement(annotation, "filename").text = xml_filename
    ET.SubElement(annotation, "path").text = xml_path

    source = ET.Element("source")
    ET.SubElement(source, "database").text = "Unknown"
    annotation.append(source)

    size = ET.Element("size")
    ET.SubElement(size, "width").text = xml_size
    ET.SubElement(size, "height").text = xml_height
    ET.SubElement(size, "depth").text = "3"
    annotation.append(size)

    ET.SubElement(annotation, "segmented").text = "0"

    object = ET.Element("object")
    ET.SubElement(object, "name").text = "widemelon"
    ET.SubElement(object, "pose").text = "Unspecified"
    ET.SubElement(object, "truncated").text = "0"
    ET.SubElement(object, "difficult").text = "0"
    bndbox = ET.Element("bndbox")
    ET.SubElement(bndbox, "xmin").text = xml_xmin
    ET.SubElement(bndbox, "ymin").text = xml_ymin
    ET.SubElement(bndbox, "xmax").text = xml_xmax
    ET.SubElement(bndbox, "ymax").text = xml_ymax
    object.append(bndbox)
    annotation.append(object)

    tree = ET.ElementTree(annotation)
    tree.write(filename)


video_dir = 'video/'
temp_dir = 'temp/'
video_name = 'video#1.mp4'
video = video_dir + video_name

frame_size = 720
frame_height = 480
frame_index = 1
temp_video_dir = temp_dir + video_name + "/"
if not os.path.exists(temp_video_dir):
    # Создать директорию с временными файлами для заданного видео
    os.makedirs(temp_video_dir)
else:
    # Очистить все файлы в директории с временными файлами для заданного видео
    files = glob.glob(temp_video_dir + "*")
    for f in files:
        os.remove(f)

first_frame = None

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video)

# while True:
while cap.isOpened():
    ret, frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame, (frame_size, frame_height))

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

            # Сохранить кадр из видео в png-файл
            cv2.imwrite(temp_video_dir + 'frame' + str(frame_index) + '.png', image)

            # Сохранить описание кадра в xml-файл
            create_xml(temp_video_dir + 'frame' + str(frame_index) + '.xml',
                       temp_video_dir,
                       str(frame_index) + '.png',
                       temp_video_dir + 'frame' + str(frame_index) + '.png',
                       str(frame_size),
                       str(frame_height),
                       str(x),
                       str(y),
                       str(x + w),
                       str(y + h)
                       )

            frame_index = frame_index + 1

            cv2.imshow('frame-result', image)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
