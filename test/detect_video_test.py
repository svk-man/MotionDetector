"""
Sections of this code were taken from:
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
"""
import glob
from os.path import join

import numpy as np

import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

import cv2

from pascal_voc_writer import Writer

# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = 'inference_graph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../data', 'labelmap.pbtxt')

NUM_CLASSES = 2

sys.path.append("..")

def remove_files_in_dir(video_images_dir):
    images = glob.glob(join(video_images_dir, "*"))
    for f in images:
        if not os.path.isdir(f):
            os.remove(f)


def detect_in_video(video_path):

    # VideoWriter is the responsible of creating a copy of the video
    # used for the detections but with the detections overlays. Keep in
    # mind the frame size has to be the same as original video.
    #out = cv2.VideoWriter('../temp/' + 'WIN_20191218_11_03_57_Pro.mp4', cv2.VideoWriter_fourcc(
    #    'M', 'J', 'P', 'G'), 10, (1280, 720))

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        configuration = tf.ConfigProto(device_count={"GPU": 0})
        sess = tf.Session(config=configuration, graph=detection_graph)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    frame_statistics = []
    frame_id = 1
    is_skip_frame = True
    frame_skip_count = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object
            # was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class
            # label.
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            # Создать директорию с кадрами для заданного видео
            video_base_name = os.path.basename(video_path)
            video_name = os.path.splitext(video_base_name)[0]
            video_dir = join(os.path.dirname(video_path), video_name)
            images_dir = "images"
            video_images_dir = join(video_dir, images_dir)

            if not os.path.exists(video_images_dir):
                os.makedirs(video_images_dir)
            else:
                # Удалить все кадры из целевой директории
                remove_files_in_dir(video_images_dir)

            video_images_dir_rat = join(video_images_dir, 'rat')
            video_images_dir_mouse = join(video_images_dir, 'mouse')
            os.makedirs(video_images_dir_rat, exist_ok=True)
            os.makedirs(video_images_dir_mouse, exist_ok=True)
            remove_files_in_dir(video_images_dir_rat)
            remove_files_in_dir(video_images_dir_mouse)

            # Загрузка видео
            cap = cv2.VideoCapture(video_path)

            # Узнать разрешение видео
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Указать разрешение картинок

            cur_dir = os.getcwd()
            os.chdir(video_images_dir)
            while cap.isOpened():
                # Read the frame
                ret, frame = cap.read()
                if frame is not None:
                    # Recolor the frame. By default, OpenCV uses BGR color space.
                    # This short blog post explains this better:
                    # https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/
                    # color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if not is_skip_frame:
                        image_np_expanded = np.expand_dims(frame, axis=0)

                        # Actual detection.
                        (boxes, scores, classes, num) = sess.run(
                            [detection_boxes, detection_scores,
                             detection_classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})

                        # Visualization of the results of a detection.
                        # note: perform the detections using a higher threshold
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            frame,
                            np.squeeze(boxes[0]),
                            np.squeeze(classes[0]).astype(np.int32),
                            np.squeeze(scores[0]),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8,
                            max_boxes_to_draw=1,
                            min_score_thresh=.20)

                        rodent_confidence = np.squeeze(scores[0])[0]
                        rodent_class_id = np.squeeze(classes[0]).astype(np.int32)[0]
                        rodent_class_name = category_index[rodent_class_id]['name']
                        if rodent_confidence > .20:
                            frame_statistics.append({'frame_id': frame_id,
                                                     'confidence': rodent_confidence,
                                                     'rodent_class_id': rodent_class_id,
                                                     'rodent_class_name': rodent_class_name,
                                                     })

                            # Сохранить кадр
                            frame_name = rodent_class_name + '/image' + str(frame_id) + '.jpg'
                            cv2.imwrite(frame_name, frame)

                            # Сохранить xml-файл
                            scores = np.squeeze(scores[0])
                            for i in range(min(1, np.squeeze(boxes[0]).shape[0])):
                                if scores is None or scores[i] > .20:
                                    boxes = tuple(boxes[i].tolist())

                            bbox_coords = boxes[0]
                            writer = Writer('.', video_width, video_height)
                            writer.addObject(rodent_class_name, bbox_coords[1] * video_width,
                                             bbox_coords[0] * video_height, bbox_coords[3] * video_width,
                                             bbox_coords[2] * video_height)
                            writer.save('image' + str(frame_id) + '.xml')
                        else:
                            # Сохранить кадр
                            frame_name = 'image' + str(frame_id) + '.jpg'
                            cv2.imwrite(frame_name, frame)

                    cv2.imshow('frame', cv2.resize(frame, (800, 600)))
                    output_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # out.write(output_rgb

                    # Пропустить кадр, если необходимо
                    if is_skip_frame:
                        while 1:
                            key = cv2.waitKey(1)
                            if key == 32:  # Нажата клавиша "space"
                                frame_skip_count += 1
                                print("Вы пропустили " + str(frame_skip_count) + " кадр")
                                break
                            elif key == 113 or key == 233:  # Нажата клавиша 'q' ('й')
                                is_skip_frame = False
                                break

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    frame_id += 1

            #out.release()
            os.chdir(cur_dir)
            cap.release()
            cv2.destroyAllWindows()

    statistics = {
        'frame_count': frame_id,                    # Количество кадров
        'frame_skip_count': frame_skip_count,       # Количество пропущенных кадров
        'frame_rodent_count': 0,                    # Количество кадров с грызуном
        'frame_rat_count': 0,                       # Количество кадров с крысой
        'frame_mouse_count': 0,                     # Количество кадров с мышью
        'sum_confidence_rat': 0,                    # Сумма вероятностей крысы на видео
        'sum_confidence_mouse': 0,                  # Сумма вероятностей мыши на видео
        'mean_confidence_rat': 0,                   # Средняя вероятность крысы на видео
        'mean_confidence_mouse': 0                  # Средняя вероятность мыши на видео
    }

    for frame_statistic in frame_statistics:
        if frame_statistic['rodent_class_name'] == 'rat':
            statistics['frame_rodent_count'] += 1
            statistics['frame_rat_count'] += 1
            statistics['sum_confidence_rat'] += frame_statistic['confidence']
            statistics['mean_confidence_rat'] = statistics['sum_confidence_rat'] / statistics['frame_rat_count']
        elif frame_statistic['rodent_class_name'] == 'mouse':
            statistics['frame_rodent_count'] += 1
            statistics['frame_mouse_count'] += 1
            statistics['sum_confidence_mouse'] += frame_statistic['confidence']
            statistics['mean_confidence_mouse'] = statistics['sum_confidence_mouse'] / statistics['frame_mouse_count']

    print('----->>> Результаты обнаружения <<<-----')
    print('Количество кадров: ' + str(statistics['frame_count']))
    print('Количество пропущенных кадров: ' + str(statistics['frame_skip_count']))
    print('Количество кадров с грызуном: ' + str(statistics['frame_rodent_count']))
    print('Количество кадров с крысой: ' + str(statistics['frame_rat_count']))
    print('Количество кадров с мышью: ' + str(statistics['frame_mouse_count']))
    print('Средняя вероятность крысы на видео: ' + str(statistics['mean_confidence_rat']))
    print('Средняя вероятность мыши на видео: ' + str(statistics['mean_confidence_mouse']))


def main():
    detect_in_video('../temp/' + 'WIN_20191218_11_03_57_Pro.mp4')


if __name__ == '__main__':
    main()