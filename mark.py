import argparse
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

drawing = False
ix, iy = -1, -1
jx, jy = -1, -1


def draw_rect(event, x, y, flags, param):
    global drawing, ix, iy, jx, jy
    jx, jy = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(clone_image, (ix, iy), (x, y), (0, 255, 0), 2)


image = cv2.imread('temp/video#1.mp4/frame1824.png')
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rect)
clone_image = image.copy()

while 1:
    if drawing:
        cv2.rectangle(clone_image, (ix, iy), (jx, jy), (0, 255, 0), 2)

    cv2.imshow("image", clone_image)
    if drawing:
        clone_image = image.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
