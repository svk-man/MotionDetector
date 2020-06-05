"""
Sections of this code were taken from:
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
"""
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

# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = 'inference_graph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../data', 'labelmap.pbtxt')

NUM_CLASSES = 2

sys.path.append("..")


def detect_in_video():

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
            cap = cv2.VideoCapture('../temp/' + 'WIN_20191218_11_03_57_Pro.mp4')
            i = 0
            while cap.isOpened():
                # Read the frame
                ret, frame = cap.read()
                i = i + 1

                # Recolor the frame. By default, OpenCV uses BGR color space.
                # This short blog post explains this better:
                # https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/
                color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image_np_expanded = np.expand_dims(color_frame, axis=0)
                if i >= 1000 and i < 1002:
                    # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores,
                         detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # Visualization of the results of a detection.
                    # note: perform the detections using a higher threshold
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        color_frame,
                        np.squeeze(boxes[0]),
                        np.squeeze(classes[0]).astype(np.int32),
                        np.squeeze(scores[0]),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        max_boxes_to_draw=1,
                        min_score_thresh=.20)

                    print(np.squeeze(scores[0]))

                    rodent_confidence = np.squeeze(scores[0])[0]
                    rodent_class_id = np.squeeze(classes[0]).astype(np.int32)[0]
                    rodent_class_name = category_index[rodent_class_id]['name']
                    if rodent_confidence > .20:
                        frame_statistics.append({'frame_id': frame_id,
                                           'confidence': rodent_confidence,
                                           'rodent_class_id': rodent_class_id,
                                           'rodent_class_name': rodent_class_name,
                                           })

                cv2.imshow('frame', cv2.resize(color_frame, (800, 600)))
                output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                #out.write(output_rgb)
                frame_id += 1

                if i == 1010:
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            #out.release()
            cap.release()
            cv2.destroyAllWindows()

    statistics = {
        'frame_count': frame_id,    # Количество кадров
        'frame_skip_count': 0,      # Количество пропущенных кадров
        'frame_rodent_count': 0,    # Количество кадров с грызуном
        'frame_rat_count': 0,       # Количество кадров с крысой
        'frame_mouse_count': 0,     # Количество кадров с мышью
        'sum_confidence_rat': 0,    # Сумма вероятностей крысы на видео
        'sum_confidence_mouse': 0,  # Сумма вероятностей мыши на видео
        'mean_confidence_rat': 0,   # Средняя вероятность крысы на видео
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
    print('Количество кадров с грызуном: ' + str(statistics['frame_rodent_count']))
    print('Количество кадров с крысой: ' + str(statistics['frame_rat_count']))
    print('Количество кадров с мышью: ' + str(statistics['frame_mouse_count']))
    print('Средняя вероятность крысы на видео: ' + str(statistics['mean_confidence_rat']))
    print('Средняя вероятность мыши на видео: ' + str(statistics['mean_confidence_mouse']))


def main():
    detect_in_video()


if __name__ == '__main__':
    main()