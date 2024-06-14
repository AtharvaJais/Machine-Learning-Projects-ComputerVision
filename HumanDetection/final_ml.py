# Importing the Libraries
import cv2
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from object_detection.utils import label_map_util

# Path to frozen inference graph and label map
model_path = '/home/atharva/Documents/ComputerVision/HumanDetection/frozen_inference_graph.pb'
label_map_path = '/home/atharva/Documents/ComputerVision/HumanDetection/label_map.pbtxt'

# Load the frozen TensorFlow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    graph_def = tf.compat.v1.GraphDef()

    with tf.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

# Load the label map
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
video_capture = cv2.VideoCapture(0)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, frame = video_capture.read()
            frame_resized = cv2.resize(frame, (800, 600))
            frame_expanded = np.expand_dims(frame_resized, axis=0)

            # Get input and output tensors from the detection graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            humans = detection_graph.get_tensor_by_name('detection_humans:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Perform Human detection
            (humans, scores, classes, num) = sess.run([humans, scores, classes, num_detections], 
                                                      feed_dict={image_tensor: frame_expanded})

            unique_values_array = np.array([])
            unique_values, counts = np.unique(unique_values_array, return_counts=True)