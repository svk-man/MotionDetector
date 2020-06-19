from __future__ import division, print_function
import cv2
import sys

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer, Qt, QPoint
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog
import cv2

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import time
from pascal_voc_writer import Writer
from os.path import join
import glob
import csv

from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QLabel, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import qimage2ndarray

import os

if os.name == "nt":  # if windows
    from PyQt5 import __file__

    pyqt_plugins = os.path.join(os.path.dirname(__file__), "Qt", "plugins")
    QApplication.addLibraryPath(pyqt_plugins)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugins

BASE_DIR = os.path.dirname(__file__)
path = BASE_DIR.replace('\\'[0], '/')


class VideoPlayer(QMainWindow):

    def __init__(self, width=640, height=640, fps=45):
        super(VideoPlayer, self).__init__()
        loadUi('view.ui', self)
        self.setWindowTitle('MotionDetector')

        self.fps = fps

        self.width = width
        self.height = height

        self.display.setMinimumSize(self.width, self.height)

        # create a timer
        self.timer = QTimer()

        # set timer timeout callback function
        self.timer.timeout.connect(self.playVideo)

        # set control_bt callback clicked function
        self.playButton.clicked.connect(self.playTimer)
        self.stopButton.clicked.connect(self.stopTimer)
        self.browseButton.clicked.connect(self.openFile)
        self.detectButton.clicked.connect(self.detectVideo)
        self.gotoButton.clicked.connect(self.jumpVideo)
        self.slider.valueChanged.connect(self.skipFrame)
        left = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self)
        left.activated.connect(self.skipLeft)
        right = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self)
        right.activated.connect(self.skipRight)
        up = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Up), self)
        up.activated.connect(self.skipUp)
        down = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Down), self)
        down.activated.connect(self.skipDown)
        space = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self)
        space.activated.connect(self.space)
        open = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+O"), self)
        open.activated.connect(self.openFile)

        # set enabled
        self.playButton.setEnabled(False)
        self.stopButton.setEnabled(False)
        self.detectButton.setEnabled(False)
        self.gotoButton.setEnabled(False)
        self.slider.setEnabled(False)

        self.detector = Detector()
        self.save_dir = 'output/'
        self.save_images_dir = 'images/'
        self.save_images_dir_rat = 'rat/'
        self.save_images_dir_mouse = 'mouse/'

    def jumpVideo(self):
        jump = int(self.gotoLine.text())
        self.cap.set(1, jump)
        self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        self.cap.set(1, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.playVideo()

    def skipUp(self):
        self.cap.set(1, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) + 999)
        self.playVideo()

    def skipDown(self):
        self.cap.set(1, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1001)
        self.playVideo()

    def skipLeft(self):
        self.cap.set(1, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 2)
        ret, image = self.cap.read()
        progress = str(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))) + ' / ' \
                   + str(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.progresslabel.setText(progress)
        self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize image
        image = cv2.resize(image, (self.width, self.height))

        # get image infos
        height, width, channel = image.shape
        step = channel * width

        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)

        # show image in img_label
        self.display.setPixmap(QPixmap.fromImage(qImg))
        self.cap.set(1, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.playVideo()

    def skipRight(self):
        self.cap.set(1, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        self.playVideo()

    def space(self):
        if self.timer.isActive():
            self.stopTimer()
        else:
            self.playTimer()

    def detectVideo(self):

        # Создать директорию с кадрами для заданного видео
        video_base_name = os.path.basename(self.file_name)
        video_name = os.path.splitext(video_base_name)[0]
        video_dir = self.save_dir + video_name + '/'
        video_images_dir = video_dir + self.save_images_dir

        if not os.path.exists(video_images_dir):
            os.makedirs(video_images_dir)
        else:
            # Удалить все кадры из целевой директории
            self.remove_files_in_dir(video_images_dir)

        video_images_dir_rat = video_images_dir + self.save_images_dir_rat
        video_images_dir_mouse = video_images_dir + self.save_images_dir_mouse
        os.makedirs(video_images_dir_rat, exist_ok=True)
        os.makedirs(video_images_dir_mouse, exist_ok=True)
        self.remove_files_in_dir(video_images_dir_rat)
        self.remove_files_in_dir(video_images_dir_mouse)

        # video properties
        video = cv2.VideoCapture(self.file_name)
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = video.get(cv2.CAP_PROP_FPS)

        # out video properties
        file_out_name = video_dir + video_name + '_detected.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_out_name, fourcc, input_fps, (self.width, self.height))

        # Сформировать заголовок csv-файла с разметкой
        images_list = [['image_id', 'x', 'y', 'w', 'h']]

        frame_statistics = []
        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        frame_skip_count = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_id >= 0 & frame_id <= total_frames:
            video.set(1, frame_id)
        points = []
        frame_points = None
        while True:
            ret, frame = video.read()
            if ret is True:
                frame_id += 1
                frame_detect = frame.copy()
                frame_detect = cv2.cvtColor(frame_detect, cv2.COLOR_BGR2RGB)
                frame_detect = cv2.resize(frame_detect, (self.width, self.height))
                frame_points = frame_detect.copy()
                frame_info = self.detector.detect(self.detector.session, frame_detect)
                frame_id = int(video.get(cv2.CAP_PROP_POS_FRAMES))

                # get statistics if rodent is found
                rodent_confidence = np.squeeze(frame_info['scores'][0])[0]
                rodent_class_id = np.squeeze(frame_info['classes'][0]).astype(np.int32)[0]
                rodent_class_name = self.detector.category_index[rodent_class_id]['name']
                if rodent_confidence > self.detector.min_score_thresh:
                    frame_statistics.append({'frame_id': frame_id,
                                             'confidence': rodent_confidence,
                                             'rodent_class_id': rodent_class_id,
                                             'rodent_class_name': rodent_class_name,
                                             })

                    # save frame
                    frame_name = video_images_dir + rodent_class_name + '/image_' + str(frame_id) + '.png'
                    cv2.imwrite(frame_name, frame)

                    # save xml-file
                    scores = np.squeeze(frame_info['scores'][0])
                    for i in range(min(1, np.squeeze(frame_info['boxes'][0]).shape[0])):
                        if scores is None or scores[i] > self.detector.min_score_thresh:
                            boxes = tuple(frame_info['boxes'][i].tolist())

                    bbox_coords = boxes[0]
                    writer = Writer(frame_name, video_width, video_height)
                    x_min, y_min = bbox_coords[1] * video_width, bbox_coords[0] * video_height,
                    x_max, y_max = bbox_coords[3] * video_width, bbox_coords[2] * video_height
                    writer.addObject(rodent_class_name, x_min, y_min, x_max, y_max)
                    writer.save(video_images_dir + rodent_class_name + '/image_' + str(frame_id) + '.xml')

                    # save csv-file
                    images_list.append([frame_name,  # q
                                        x_min, y_max,  # 'x', 'y'
                                        x_max - x_min, y_max - y_min])  # 'w', 'h'

                    # save track points
                    x_min_sm, y_min_sm = bbox_coords[1] * self.width, bbox_coords[0] * self.height,
                    x_max_sm, y_max_sm = bbox_coords[3] * self.width, bbox_coords[2] * self.height
                    point_x = (x_min_sm + (x_max_sm - x_min_sm) / 2)
                    point_y = (y_min_sm + (y_max_sm - y_min_sm) / 2)
                    points.append([point_x, point_y])  # 'w', 'h'
                else:
                    # save frame
                    frame_name = video_images_dir + '/image_' + str(frame_id) + '.png'
                    cv2.imwrite(frame_name, frame)

                frame_detect = cv2.cvtColor(frame_detect, cv2.COLOR_RGB2BGR)
                out.write(frame_detect)
            else:
                break

        out.release()

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
                statistics['mean_confidence_mouse'] = statistics['sum_confidence_mouse'] / statistics[
                    'frame_mouse_count']

        print('----->>> Результаты обнаружения <<<-----')
        print('Количество кадров: ' + str(statistics['frame_count']))
        print('Количество пропущенных кадров: ' + str(statistics['frame_skip_count']))
        print('Количество кадров с грызуном: ' + str(statistics['frame_rodent_count']))
        print('Количество кадров с крысой: ' + str(statistics['frame_rat_count']))
        print('Количество кадров с мышью: ' + str(statistics['frame_mouse_count']))
        print('Средняя вероятность крысы на видео: ' + str(statistics['mean_confidence_rat']))
        print('Средняя вероятность мыши на видео: ' + str(statistics['mean_confidence_mouse']))

        # Создать CSV-файл с разметкой и записать в него данные
        csv_file_name = 'mark.csv'
        with open(video_dir + csv_file_name, mode='w', newline='') as csv_file:
            csv_file_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_NONE)
            csv_file_writer.writerows(images_list)

        if points:
            self.open_new_dialog(frame_points, points)

        self.browseButton.setEnabled(True)
        self.playButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.slider.setEnabled(True)
        self.gotoButton.setEnabled(True)

    def remove_files_in_dir(self, video_images_dir):
        images = glob.glob(join(video_images_dir, "*"))
        for f in images:
            if not os.path.isdir(f):
                os.remove(f)

    def exportVideo(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        write_video = cv2.VideoCapture(self.file_name)
        input_fps = write_video.get(cv2.CAP_PROP_FPS)
        file = self.file_name.split('.')[0]
        file_name = file + '_converted_.mp4'
        print(file_name)
        out = cv2.VideoWriter(file_name, fourcc, input_fps, (int(write_video.get(3)), int(write_video.get(4))))
        while write_video.isOpened():
            ret, frame = write_video.read()
            if ret is True:
                print(write_video.get(cv2.CAP_PROP_POS_FRAMES))
                out.write(frame)
            else:
                break
        out.release()

    def skipFrame(self):
        value = self.slider.value()
        self.cap.set(1, value)
        progress = str(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))) + ' / ' \
                   + str(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.progresslabel.setText(progress)

    def playVideo(self):
        # read image in BGR format
        ret, image = self.cap.read()
        if ret is True:
            progress = str(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))) + ' / ' \
                       + str(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.progresslabel.setText(progress)
            self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            # convert image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # resize image
            image = cv2.resize(image, (self.width, self.height))
            # detect
            frame_info = self.detector.detect(self.detector.session, image)
            image = frame_info['frame']
            # get image infos
            height, width, channel = image.shape
            step = channel * width
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            self.display.setPixmap(QPixmap.fromImage(qImg))
        else:
            progress = str(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) + ' / ' \
                       + str(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.progresslabel.setText(progress)
            self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            self.stopTimer()

    def openFile(self):
        self.browseButton.setEnabled(False)
        self.videoFileName = QFileDialog.getOpenFileName(self, 'Select Video File')
        self.file_name = list(self.videoFileName)[0]
        self.cap = cv2.VideoCapture(self.file_name)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        # set enabled
        self.playButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.detectButton.setEnabled(True)
        self.gotoButton.setEnabled(True)
        self.slider.setEnabled(True)

        self.playVideo()
        self.browseButton.setEnabled(True)

    def playTimer(self):
        # start timer
        self.timer.start(1000 // self.fps)

        # set enabled
        self.playButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.browseButton.setEnabled(False)
        self.detectButton.setEnabled(False)

    def stopTimer(self):
        # stop timer
        self.timer.stop()

        # set enabled
        self.playButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.browseButton.setEnabled(True)
        self.detectButton.setEnabled(True)

    def close_win(self):
        cv2.destroyAllWindows()
        self.close()

    def open_new_dialog(self, frame, points):
        self.SW = TrackView(frame, points)
        self.SW.resize(300, 300)
        self.SW.show()

class TrackView(QMainWindow):
    def __init__(self, frame, points):
        super(TrackView, self).__init__()
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        layout = QtWidgets.QVBoxLayout(self.main_widget)

        self.frame = frame
        self.points = points

        height, width, channel = self.frame.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(self.frame.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label

        label = QLabel(self)
        layout.addWidget(label)

        pixmap_image = QPixmap.fromImage(qImg)

        # create painter instance with pixmap
        painterInstance = QtGui.QPainter(pixmap_image)

        # set rectangle color and thickness
        pen = QtGui.QPen(QtCore.Qt.red)
        pen.setWidth(3)

        # draw rectangle on painter
        painterInstance.setPen(pen)
        for i in range(len(self.points)):
            if i + 1 < len(self.points):
                painterInstance.drawLine(self.points[i][0], self.points[i][1], self.points[i + 1][0],
                                         self.points[i + 1][1])

        # set pixmap onto the label widget
        label.setPixmap(pixmap_image)
        label.show()

class Detector:

    def __init__(self):
        self.detection_graph = tf.Graph()
        self.session = None
        self.path_to_ckpt = 'models/inference_graph/frozen_inference_graph.pb'
        self.path_to_labels = 'data/labelmap.pbtxt'
        self.num_classes = 2
        self.max_boxes_to_draw = 1
        self.min_score_thresh = .15

        # Get category index
        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        # Load a (frozen) Tensorflow model into memory.
        with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='prefix')
        self.detection_graph = graph
        with self.detection_graph.as_default():
            self.session = tf.Session()

    def __del__(self):
        if self.session is not None:
            self.session.close()

    def detect(self, sess, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        # Extract image tensor
        image_tensor = self.detection_graph.get_tensor_by_name('prefix/image_tensor:0')
        # Extract detection boxes
        boxes = self.detection_graph.get_tensor_by_name('prefix/detection_boxes:0')
        # Extract detection scores
        scores = self.detection_graph.get_tensor_by_name('prefix/detection_scores:0')
        # Extract detection classes
        classes = self.detection_graph.get_tensor_by_name('prefix/detection_classes:0')
        # Extract number of detections
        num_detections = self.detection_graph.get_tensor_by_name('prefix/num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        end_time = time.time()

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes[0]),
            np.squeeze(classes[0]).astype(np.int32),
            np.squeeze(scores[0]),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            max_boxes_to_draw=self.max_boxes_to_draw,
            min_score_thresh=self.min_score_thresh)

        cv2.putText(image, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                    fontScale=1, color=(0, 255, 0), thickness=2)

        image_info = {
            'frame': image,
            'boxes': boxes,
            'classes': classes,
            'scores': scores
        }

        return image_info


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
