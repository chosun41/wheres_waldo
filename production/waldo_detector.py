import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
import streamlit as st
st.write(st.config.get_option("server.enableCORS"))
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as patches
from PIL import Image
from io import BytesIO
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Waldo Object Detection")

label_map = label_map_util.load_labelmap('labels.txt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# load frozen inference graph
@st.cache(allow_output_mutation=True)
def load_model():
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('frozen_inference_graph.pb', 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return detection_graph

with st.spinner('Loading Model Into Memory....'):
  detection_graph = load_model()

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  # original size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

img_file_buffer = st.file_uploader("Upload Image to Detect Waldo....")

# display image with box for waldo
if img_file_buffer is not None:
  with st.spinner('Searching for Waldo in image ...'):
    with detection_graph.as_default():
      with tf.compat.v1.Session(graph=detection_graph) as sess:
        orig_image = img_file_buffer.getvalue()
        image_np = load_image_into_numpy_array(Image.open(img_file_buffer))
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True, # allows boxes to conform to plot figure
          line_thickness=8)
        # control size of figure
        plt.figure(figsize=(24,16))
        st.image(image_np)
        st.write("Image Uploaded Successfully")
        
        if scores[0][0]<0.5:
          st.write("Waldo Not Detected!")
        else:
          st.write("Waldo Detected!")
          norm_box=boxes[0][0]
          im_width,im_height=image_np.shape[0],image_np.shape[1]
          st.write(f"The original size of the image is {image_np.shape}")
          
          box = (int(norm_box[1]*im_height),int(norm_box[0]*im_width),int(norm_box[3]*im_height),int(norm_box[2]*im_width))
          st.write(f"The coordinates of waldo in the original image is {box} - (xmin,ymin,xmax,ymax)")
          
          cropped_image = image_np[box[1]:box[3],box[0]:box[2],:]
          st.write(f"The size of the cropped image is {cropped_image.shape}")
          st.image(cropped_image)
