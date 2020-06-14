from __future__ import division, print_function
import cv2
import sys

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog
import cv2

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import time

import os
if os.name == "nt":  # if windows
    from PyQt5 import __file__
    pyqt_plugins = os.path.join(os.path.dirname(__file__), "Qt", "plugins")
    QApplication.addLibraryPath(pyqt_plugins)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugins

BASE_DIR = os.path.dirname(__file__)
path = BASE_DIR.replace('\\'[0], '/')


class VideoPlayer(QMainWindow):

    pause = False
    video = False

    def __init__(self, width=640, height=640, custom_fps=60):
        super(VideoPlayer, self).__init__()
        loadUi('view.ui', self)
        self.setWindowTitle('MotionDetector')

        # create a timer
        self.timer = QTimer()

        # set timer timeout callback function
        self.timer.timeout.connect(self.playVideo)

        # set control_bt callback clicked function
        self.playButton.clicked.connect(self.playTimer)
        self.stopButton.clicked.connect(self.stopTimer)
        self.browseButton.clicked.connect(self.openFile)
        self.exportButton.clicked.connect(self.convertVideo)
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

        self.detector = Detector()

    def jumpVideo(self):
        jump = int(self.gotoLine.text())
        self.cap.set(1, jump)
        self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        self.cap.set(1, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.playVideo()

    def skipUp(self):
        self.cap.set(1, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))+999)
        self.playVideo()

    def skipDown(self):
        self.cap.set(1, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1001)
        self.playVideo()

    def skipLeft(self):
        self.cap.set(1, int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-2)
        ret, image = self.cap.read()
        progress = str(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))) + ' / ' \
                   + str(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.progresslabel.setText(progress)
        self.slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))

        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize image
        image = cv2.resize(image, (640, 640))

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
            self.timer.stop()
        else:
            self.timer.start()

    def convertVideo(self):
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

    def exportVideo(self):
        start = self.startline.text()
        end = self.endline.text()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        write_video = cv2.VideoCapture(self.file_name)
        input_fps = write_video.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(path+'/export_left_'+start+'_'+end+'.mp4', fourcc, input_fps, (360, 240))
        write_video.set(1, int(start)-1)
        for cur in range(int(end)-int(start)+1):
            ret, frame = write_video.read()
            progress = str(int(write_video.get(cv2.CAP_PROP_POS_FRAMES))) + ' / ' \
                       + str(int(end))
            self.exportLabel.setText(progress)
            print(progress)
            image = frame[240:480, 0:360]
            out.write(image)
        out.release()

    def skipFrame(self):
        value = self.slider.value()
        self.cap.set(1, value)

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
            image = cv2.resize(image, (640, 640))
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
        self.videoFileName = QFileDialog.getOpenFileName(self, 'Select Video File')
        self.file_name = list(self.videoFileName)[0]
        self.cap = cv2.VideoCapture(self.file_name)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        #self.timer.start(fps)

    def playTimer(self):
        # start timer
        self.timer.start(20)

    def stopTimer(self):
        # stop timer
        self.timer.stop()

    def close_win(self):
        cv2.destroyAllWindows()
        self.close()


class Detector:

    def __init__(self):
        self.detection_graph = tf.Graph()
        self.session = None
        self.path_to_ckpt = 'models/inference_graph/frozen_inference_graph.pb'
        self.path_to_labels = 'data/labelmap.pbtxt'
        self.num_classes = 2
        self.max_boxes_to_draw = 1
        self.min_score_thresh = .20

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
        # Extract number of detectionsd
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

        return image


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
