import argparse
import csv
import glob
import os
import sys

import cv2

'''def validate(video_path, n):
    message = ''
    if not os.path.isfile(video_path):
        message = 'Указанный путь к видео не существует'

    if n <= 0:
        message = 'N не может быть отрицательным числом'

    return message


parser_desc = 'Консольная утилита для разметки видео для проекта с мыше-крысами. Язык - Python 3.7+, основная либа - ' \
             'OpenCV.\n' \
             'пример вызова:' \
             'python mark.py "video/clip1.avi" 10'

parser = argparse.ArgumentParser(prog='mark',
                                 description=parser_desc)

parser.add_argument('Path',
                    metavar='path',
                    type=str,
                    help='путь к видео')
parser.add_argument('N',
                    metavar='n',
                    type=int,
                    nargs='?',
                    help='использовать каждый N-ый кадр, то есть параметр - целое число N (например, N = 5, '
                         'это значит, что обрабатывать нужно каждый 5-ый кадр, а 4 пропускать. По умолчанию N = 10',
                    default=10)

args = parser.parse_args()
video_path = args.Path
n = args.N

error_message = validate(video_path, n)
if error_message:
    print(error_message)
    sys.exit()

print(video_path)
print(n)'''


def draw_rect(event, x, y, flags, param):
    global drawing, draging, ix, iy, jx, jy, dx1, dy1, dx2, dy2, frame, clone_frame, rect
    jx, jy = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        if (not draging) and (rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]):
            drawing = True
            draging = True
            dx1, dy1 = x, y
        else:
            drawing = True
            ix, iy = x, y

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            clone_frame = frame.copy()
            if not draging:
                cv2.rectangle(clone_frame, (ix, iy), (jx, jy), (0, 255, 0), 2)
            else:
                cv2.rectangle(clone_frame, (rect[0] + x - dx1, rect[1] + y - dy1),
                              (rect[2] + x - dx1, rect[3] + y - dy1), (0, 255, 0), 2)

    if event == cv2.EVENT_LBUTTONUP:
        if ix != -1 and iy != -1:
            clone_frame = frame.copy()
            if not draging:
                cv2.rectangle(clone_frame, (ix, iy), (jx, jy), (0, 255, 0), 2)
                rect = (ix, iy, jx, jy)
            else:
                cv2.rectangle(clone_frame, (rect[0] + x - dx1, rect[1] + y - dy1),
                              (rect[2] + x - dx1, rect[3] + y - dy1),
                              (0, 255, 0), 2)
                rect = (rect[0] + x - dx1, rect[1] + y - dy1, rect[2] + x - dx1, rect[3] + y - dy1)
            drawing = False
            draging = False


drawing = False
draging = False
ix, iy = -1, -1
jx, jy = -1, -1
dx1, dy1, dx2, dy2 = -1, -1, -1, -1
rect = (-1, -1, -1, -1)

cv2.namedWindow('frame')

frame_size = 720
frame_height = 480

N = 10

video_path = 'video/video#1.mp4'

# Создать директорию с файлами разметки для заданного видео
video_base_name = os.path.basename(video_path)
video_name = os.path.splitext(video_base_name)[0]
temp_dir = 'temp/'
temp_video_dir = temp_dir + video_name + "/"
images_dir = "images/"

if not os.path.exists(temp_video_dir + images_dir):
    os.makedirs(temp_video_dir + images_dir)
else:
    # Очистить все файлы в директориях
    files = glob.glob(temp_video_dir + "*")
    for f in files:
        if not os.path.isdir(f):
            os.remove(f)

    files = glob.glob(temp_video_dir + images_dir + "*")
    for f in files:
        if not os.path.isdir(f):
            os.remove(f)

# Загрузка видео
cap = cv2.VideoCapture(video_path)

# Сформировать заголовок csv-файла с разметкой
images_list = [['image_id', 'x', 'y', 'w', 'h', 'x+w', 'y+h']]

is_quit = False
is_first_frame = True
i = 0
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if frame is not None:
        i += 1
        if not (i % N):
            frame = cv2.resize(frame, (frame_size, frame_height))

            clone_frame = frame.copy()
            cv2.setMouseCallback('frame', draw_rect)
            is_first_frame = False
            if rect[2] != -1 and rect[3] != -1:
                cv2.rectangle(clone_frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

            while 1:
                cv2.imshow('frame', clone_frame)

                key = cv2.waitKey(1)
                if key == 113 or key == 233:  # Нажата клавиша 'q' ('й')
                    is_quit = True
                    break
                elif key == 32:  # Нажата клавиша "space"
                    break
                elif key == 27:  # Нажата клавиша "ESC"
                    clone_frame = frame.copy()
                    cv2.imshow('frame', clone_frame)
                    drawing = False
                    draging = False
                    ix, iy = -1, -1
                    jx, jy = -1, -1
                    dx1, dy1, dx2, dy2 = -1, -1, -1, -1
                    rect = (-1, -1, -1, -1)

            if rect[2] != -1 and rect[3] != -1:
                # Сохранить размеченный кадр в jpg-файл
                frame_id += 1
                frame_name = 'image' + str(frame_id) + '.jpg'
                cv2.imwrite(temp_video_dir + images_dir + frame_name, clone_frame)

                # Сохранить информацию о размеченном кадре для csv-файла
                images_list.append([images_dir + frame_name,  # 'image_id'
                                    rect[0], rect[1],  # 'x', 'y'
                                    rect[2] - rect[0], rect[3] - rect[1],  # 'w', 'h'
                                    rect[2], rect[3]])  # 'x+w', 'y+h'

    if is_quit:
        break

cap.release()
cv2.destroyAllWindows()

# Создать CSV-файл с разметкой и записать в него данные
csv_file_name = 'mark.csv'
with open(temp_video_dir + csv_file_name, mode='w', newline='') as csv_file:
    csv_file_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE)
    csv_file_writer.writerows(images_list)
