import argparse
import glob
import os
import sys
import cv2


def validate(video_path, n):
    message = ''
    if not os.path.isfile(video_path):
        message = 'Указанный путь к видео не существует'

    if n <= 0:
        message = 'N не может быть отрицательным числом'

    return message


parser_desc = 'Консольная утилита для разделения видео на кадры для проекта с мыше-крысами. Язык - Python 3.7+, ' \
              'основная либа - ' \
              'OpenCV.\n' \
              'пример вызова:' \
              'python split.py "video/clip1.avi" 10'

parser = argparse.ArgumentParser(prog='split',
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


def split_video(video_path, n):
    cv2.namedWindow('frame')

    # Создать директорию с кадрами для заданного видео
    video_base_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_base_name)[0]
    video_dir = video_name + "/"
    images_dir = "images/"
    video_images_dir = video_dir + images_dir

    if not os.path.exists(video_images_dir):
        os.makedirs(video_images_dir)
    else:
        # Удалить все кадры из целевой директории
        images = glob.glob(video_images_dir + "*")
        for f in images:
            if not os.path.isdir(f):
                os.remove(f)

    # Загрузка видео
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            frame_id += 1
            if not (frame_id % n):
                frame_name = 'image' + str(frame_id) + '.jpg'
                cv2.imwrite(video_images_dir + frame_name, frame)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


split_video(video_path, n)
