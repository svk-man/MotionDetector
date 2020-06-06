"""
Sections of this code were taken from:
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
"""
from __future__ import division, print_function

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

import argparse
import cv2
import time

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

print(tf.version.VERSION)
# category_index = [
#     {'id': 0, 'name': 'rat'},
#     {'id': 1, 'name': 'mouse'}
# }]
is_yolo = True
if is_yolo:
    anchors = parse_anchors('../data/yolo_anchors.txt')
    classes_yolo = read_class_names('../data/data.names')
    num_class = len(classes_yolo)

    color_table = get_color_table(2)

    new_size = [416, 416]

    restore_path = 'checkpoint/model-epoch_30_step_25853_loss_0.1543_lr_0.0001'

    PATH_TO_LABELS = os.path.join('../data', 'data.names')

    is_letterbox_resize = True
    PATH_TO_LABELS = os.path.join('../data', 'labelmap.pbtxt')

else:
    # Path to frozen detection graph. This is the actual model that is used
    # for the object detection.
    PATH_TO_CKPT = 'inference_graph/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('../data', 'labelmap.pbtxt')

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

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
    # out = cv2.VideoWriter('../temp/' + 'WIN_20191218_11_03_57_Pro.mp4', cv2.VideoWriter_fourcc(
    #    'M', 'J', 'P', 'G'), 10, (1280, 720))

    if is_yolo:
        print('yolo!')
        configuration = tf.ConfigProto(device_count={"GPU": 0})
        sess = tf.Session(config=configuration)
        input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
        yolo_model = yolov3(num_class, anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=1, score_thresh=0.2,
                                        nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, restore_path)
    else:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            configuration = tf.ConfigProto(device_count={"GPU": 0})
            sess = tf.Session(config=configuration, graph=detection_graph)

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

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

    frame_statistics = []
    frame_id = 1
    is_skip_frame = True
    frame_skip_count = 0

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
    video_frame_cnt = int(cap.get(7))
    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    video_fps = int(cap.get(5))

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
                if is_yolo:
                    print('yoloo!!')
                    if is_letterbox_resize:
                        img, resize_ratio, dw, dh = letterbox_resize(frame, new_size[0], new_size[1])
                    else:
                        height_ori, width_ori = frame.shape[:2]
                        img = cv2.resize(frame, tuple(new_size))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = np.asarray(img, np.float32)
                    img = img[np.newaxis, :] / 255.

                    start_time = time.time()
                    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
                    end_time = time.time()

                    # rescale the coordinates to the original image
                    if is_letterbox_resize:
                        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
                    else:
                        boxes_[:, [0, 2]] *= (width_ori / float(new_size[0]))
                        boxes_[:, [1, 3]] *= (height_ori / float(new_size[1]))

                    for i in range(len(boxes_)):
                        if scores_[i] == max(scores_):
                            x0, y0, x1, y1 = boxes_[i]
                            plot_one_box(frame, [x0, y0, x1, y1],
                                         label=classes_yolo[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                                         color=color_table[labels_[i]])

                            rodent_confidence = scores_[i]
                            rodent_class_id = labels_[i] + 1
                            rodent_class_name = classes_yolo[labels_[i]]
                            if rodent_confidence >= .20:
                                 frame_statistics.append({'frame_id': frame_id,
                                                          'confidence': rodent_confidence,
                                                          'rodent_class_id': rodent_class_id,
                                                          'rodent_class_name': rodent_class_name,
                                                          })

                                 # Сохранить кадр
                                 frame_name = rodent_class_name + '/image' + str(frame_id) + '.jpg'
                                 cv2.imwrite(frame_name, frame)

                                 # Сохранить xml-файл
                                 #scores = np.squeeze(scores[0])

                                 #bbox_coords = boxes[0]
                                 #writer = Writer('.', video_width, video_height)
                                 #writer.addObject(rodent_class_name, bbox_coords[1] * video_width,
                                                  #bbox_coords[0] * video_height, bbox_coords[3] * video_width,
                                                  #bbox_coords[2] * video_height)
                                 #writer.save('image' + str(frame_id) + '.xml')


                             #else:
                                 # Сохранить кадр
                                 #frame_name = 'image' + str(frame_id) + '.jpg'
                                 #cv2.imwrite(frame_name, frame)

                    cv2.putText(frame, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                                    fontScale=1, color=(0, 255, 0), thickness=2)

                else:
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

                # rodent_confidence = np.squeeze(scores[0])[0]
                # rodent_class_id = np.squeeze(classes[0]).astype(np.int32)[0]
                # rodent_class_name = category_index[rodent_class_id]['name']
                # if rodent_confidence > .20:
                #     frame_statistics.append({'frame_id': frame_id,
                #                              'confidence': rodent_confidence,
                #                              'rodent_class_id': rodent_class_id,
                #                              'rodent_class_name': rodent_class_name,
                #                              })
                #
                #     # Сохранить кадр
                #     frame_name = rodent_class_name + '/image' + str(frame_id) + '.jpg'
                #     cv2.imwrite(frame_name, frame)
                #
                #     # Сохранить xml-файл
                #     scores = np.squeeze(scores[0])
                #     for i in range(min(1, np.squeeze(boxes[0]).shape[0])):
                #         if scores is None or scores[i] > .20:
                #             boxes = tuple(boxes[i].tolist())
                #
                #     bbox_coords = boxes[0]
                #     writer = Writer('.', video_width, video_height)
                #     writer.addObject(rodent_class_name, bbox_coords[1] * video_width,
                #                      bbox_coords[0] * video_height, bbox_coords[3] * video_width,
                #                      bbox_coords[2] * video_height)
                #     writer.save('image' + str(frame_id) + '.xml')
                # else:
                #     # Сохранить кадр
                #     frame_name = 'image' + str(frame_id) + '.jpg'
                #     cv2.imwrite(frame_name, frame)

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

    # out.release()
    os.chdir(cur_dir)
    cap.release()
    cv2.destroyAllWindows()

    statistics = {
        'frame_count': frame_id,  # Количество кадров
        'frame_skip_count': frame_skip_count,  # Количество пропущенных кадров
        'frame_rodent_count': 0,  # Количество кадров с грызуном
        'frame_rat_count': 0,  # Количество кадров с крысой
        'frame_mouse_count': 0,  # Количество кадров с мышью
        'sum_confidence_rat': 0,  # Сумма вероятностей крысы на видео
        'sum_confidence_mouse': 0,  # Сумма вероятностей мыши на видео
        'mean_confidence_rat': 0,  # Средняя вероятность крысы на видео
        'mean_confidence_mouse': 0  # Средняя вероятность мыши на видео
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
    detect_in_video('' + 'WIN_20191218_11_26_01_Pro.mp4')


if __name__ == '__main__':
    main()