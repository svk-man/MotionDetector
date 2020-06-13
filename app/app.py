from __future__ import division, print_function
import cv2
import sys

import os
import PySide2
from PySide2 import QtCore, QtGui, QtWidgets
import qimage2ndarray

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
import time

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


class VideoPlayer(QtWidgets.QWidget):

    pause = False
    video = False

    def __init__(self, width=640, height=640, fps=60):
        QtWidgets.QWidget.__init__(self)
        self.video_size = QtCore.QSize(width, height)
        self.camera_capture = cv2.VideoCapture(cv2.CAP_DSHOW)
        self.video_capture = cv2.VideoCapture()

        self.frame_timer = QtCore.QTimer()
        self.setup_camera(fps)
        self.fps = fps

        self.frame_label = QtWidgets.QLabel()
        self.quit_button = QtWidgets.QPushButton("Quit")
        self.play_pause_button = QtWidgets.QPushButton("Pause")
        self.camera_video_button = QtWidgets.QPushButton("Switch to video")

        self.main_layout = QtWidgets.QGridLayout()

        self.setup_ui()

        QtCore.QObject.connect(self.play_pause_button, QtCore.SIGNAL("clicked()"), self.play_pause)
        QtCore.QObject.connect(self.camera_video_button, QtCore.SIGNAL("clicked()"), self.camera_video)

        self.detector = Detector()

    # Разместить в интерфейсе элементы
    def setup_ui(self):
        self.frame_label.setFixedSize(self.video_size)
        self.quit_button.clicked.connect(self.close_win)

        self.main_layout.addWidget(self.frame_label, 0, 0, 1, 2)
        self.main_layout.addWidget(self.play_pause_button, 1, 0, 1, 1)
        self.main_layout.addWidget(self.camera_video_button, 1, 1, 1, 1)
        self.main_layout.addWidget(self.quit_button, 2, 0, 1, 2)

        self.setLayout(self.main_layout)

    def play_pause(self):
        if not self.pause:
            self.frame_timer.stop()
            self.play_pause_button.setText("Play")
        else:
            self.frame_timer.start(int(1000 // self.fps))
            self.play_pause_button.setText("Pause")

        self.pause = not self.pause

    def camera_video(self):
        if not self.video:
            path = QtWidgets.QFileDialog.getOpenFileName(filter="Videos (*.mp4)")
            if len(path[0]):
                self.video_capture.open(path[0])
                self.camera_video_button.setText("Switch to camera")
        else:
            self.camera_video_button.setText('Switch no video')
            self.video_capture.release()

        self.video = not self.video

    # Считать значения и вывести на экран
    def setup_camera(self, fps):
        self.camera_capture.set(3, self.video_size.width())
        self.camera_capture.set(4, self.video_size.height())

        self.frame_timer.timeout.connect(self.display_video_stream)
        self.frame_timer.start(int(1000 // fps))

    # Показать изображение
    def display_video_stream(self):
        if not self.video:
            ret, frame = self.camera_capture.read()
        else:
            ret, frame = self.video_capture.read()

        if not ret:
            return False

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not self.video:
            frame = cv2.flip(frame, 1)
        else:
            frame = cv2.resize(frame, (self.video_size.width(), self.video_size.height()), interpolation=cv2.INTER_AREA)

        if self.video:
            frame = self.detector.predict(frame)

        image = qimage2ndarray.array2qimage(frame)

        self.frame_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def close_win(self):
        cv2.destroyAllWindows()
        self.camera_capture.release()
        self.video_capture.release()
        self.close()


class Detector:

    def __init__(self):
        self.path_to_ckpt = 'models/inference_graph/frozen_inference_graph.pb'
        self.path_to_labels = 'data/labelmap.pbtxt'
        self.num_classes = 2
        self.max_boxes_to_draw = 1
        self.min_score_thresh = .20

        #load graph
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=detection_graph)

            # Definite input and output Tensors for detection_graph
            self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object
            # was detected.
            self.detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class
            # label.
            self.detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            self.detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            self.num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

        # get category index
        label_map = label_map_util.load_labelmap(self.path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def predict(self, frame):
        image_np_expanded = np.expand_dims(frame, axis=0)

        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
             self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        # Visualization of the results of a detection.
        # note: perform the detections using a higher threshold
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes[0]),
            np.squeeze(classes[0]).astype(np.int32),
            np.squeeze(scores[0]),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            max_boxes_to_draw=self.max_boxes_to_draw,
            min_score_thresh=self.min_score_thresh)

        cv2.putText(frame, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,
                    fontScale=1, color=(0, 255, 0), thickness=2)

        return frame


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
