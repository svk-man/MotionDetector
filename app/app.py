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

    def __init__(self, width=640, height=640, camera_fps=60):
        QtWidgets.QWidget.__init__(self)
        self.video_size = QtCore.QSize(width, height)
        self.camera_capture = cv2.VideoCapture(cv2.CAP_DSHOW)
        self.video_capture = cv2.VideoCapture()

        self.frame_timer = QtCore.QTimer()
        self.setup_camera(camera_fps)
        self.fps = None

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
                # Find OpenCV version
                (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

                if int(major_ver) < 3:
                    fps = self.video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
                else :
                    fps = self.video_capture.get(cv2.CAP_PROP_FPS)
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
            frame = self.detector.detect(self.detector.session, image=frame)

        image = qimage2ndarray.array2qimage(frame)

        self.frame_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def close_win(self):
        cv2.destroyAllWindows()
        self.camera_capture.release()
        self.video_capture.release()
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
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
