# Importing the Libraries
import cv2
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from object_detection.utils import label_map_util

# Connecting to Cluster
# Path to frozen inference graph and label map
model_path = '/home/atharva/Documents/ComputerVision/demo_ml/frozen_inference_graph.pb'
label_map_path = '/home/atharva/Documents/ComputerVision/demo_ml/label_map.pbtxt'

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

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam, or specify video file path


# Defining run time
start_time = time.time()
# Defining Insertion query
ml_log_agg_query = ''
ml_log_raw_query = ''


with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            # Read a frame from the video feed
            ret, frame = video_capture.read()

            # Resize frame for improved performance (optional)
            frame_resized = cv2.resize(frame, (800, 600))

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            frame_expanded = np.expand_dims(frame_resized, axis=0)

            # Get input and output tensors from the detection graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Perform object detection
            (boxes, scores, classes, num) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: frame_expanded})


            unique_values_array = np.array([])
            unique_values, counts = np.unique(unique_values_array, return_counts=True)

            # inserting into table
            if time.time() - start_time >= 10:
                ml_log_agg_query = """INSERT into senselive_atomic.ml_log_agg (object, object_count, event_timestamp) values """
                ml_log_raw_query = """INSERT into senselive_atomic.ml_log_raw (object, event_timestamp) values """

                for i in range(int(num[0])):
                    insertion_time = datetime.now()
                    # Declaring time variable
                    if scores[0][i] > 0.5:
                        class_name = category_index[classes[0][i]]['name']
                        unique_values_array = np.append(unique_values_array, class_name)
                            
                start_time = time.time()
                unique_values, counts = np.unique(unique_values_array, return_counts=True)

                
                try:
                    i = 0
                    for unique_val, count in zip(unique_values, counts):
                        if i == 0:
                            ml_log_agg_query += f" ('{unique_val}', {count}, (toDateTime(parseDateTimeBestEffort('{insertion_time}', 'Asia/Kolkata'))))"
                        else:
                            ml_log_agg_query += f", ('{unique_val}', {count}, (toDateTime(parseDateTimeBestEffort('{insertion_time}', 'Asia/Kolkata'))))"
                        i+=1

                    i = 0
                    for val in unique_values_array:
                        if i == 0:
                            ml_log_raw_query += f" ('{val}', (toDateTime(parseDateTimeBestEffort('{insertion_time}', 'Asia/Kolkata'))))"
                        else:
                            ml_log_raw_query += f", ('{val}', (toDateTime(parseDateTimeBestEffort('{insertion_time}', 'Asia/Kolkata'))))"

                        i+= 0
                    try:
                        cluster_driver.execute(ml_log_raw_query)
                    except:
                        print(f'raw query: \n{ml_log_raw_query}')
                        print("Error occurred while executing the insert_raw statement:", str(e))
                    try:
                        cluster_driver.execute(ml_log_agg_query)
                    except Exception as e:
                        print("Error occurred while executing the insert_agg statement:", str(e))
                        print(f'agg query insertion: \n{ml_log_agg_query}')
                    print('Insertion successfullll')
                except clickhouse_driver.errors.Error as e:
                    print(f"An error occurred: {str(e)}")

                except:
                    print('Failed')