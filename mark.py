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


def draw_rect(event, x, y, flags, param):
    global drawing, draging, ix, iy, jx, jy, dx1, dy1, dx2, dy2, image, clone_image, rect
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
            clone_image = image.copy()
            if not draging:
                cv2.rectangle(clone_image, (ix, iy), (jx, jy), (0, 255, 0), 2)
            else:
                cv2.rectangle(clone_image, (rect[0] + x - dx1, rect[1] + y - dy1),
                              (rect[2] + x - dx1, rect[3] + y - dy1), (0, 255, 0), 2)

    if event == cv2.EVENT_LBUTTONUP:
        if ix != -1 and iy != -1:
            clone_image = image.copy()
            if not draging:
                cv2.rectangle(clone_image, (ix, iy), (jx, jy), (0, 255, 0), 2)
                rect = (ix, iy, jx, jy)
            else:
                cv2.rectangle(clone_image, (rect[0] + x - dx1, rect[1] + y - dy1),
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

image = cv2.imread('temp/video#1.mp4/frame1824.png')
cv2.namedWindow('image')
clone_image = image.copy()
cv2.setMouseCallback('image', draw_rect)

while 1:
    cv2.imshow("image", clone_image)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key == 27:                         # Нажата клавиша "ESC"
        clone_image = image.copy()
        cv2.imshow("image", clone_image)
        drawing = False
        draging = False
        ix, iy = -1, -1
        jx, jy = -1, -1
        dx1, dy1, dx2, dy2 = -1, -1, -1, -1
        rect = (-1, -1, -1, -1)


cv2.waitKey(0)
cv2.destroyAllWindows()
