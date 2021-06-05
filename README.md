# wheres_waldo

## Training

The training directory provides all the tools to train an object detection model based on where's waldo images. 

### Description of files:
1. images - folder of 36 training images
2. eval_images - folder of 12 test images  
3. annotations.csv - annotations for images in the images directory with the image name and box coordinates
4. create_tfrecords.py - script to create one's own tf records based upon annotations and waldo images
5. labels.txt - label for object to detect
6. new.config - config to replace the initial pipeline.config from the faster rcnn object detection model
7. test.tfrecord and train.tfrecord - train and test tfrecords created by create_tfrecords.py that is created by a 70-30 split of the annotations
8. wheres_waldo.ipynb - notebook run on google colab to retrain a faster rcnn object detection model on waldo images. 

### How to use:
For the easiest experience, run the wheres_waldo notebook on google colab starting from the section where it says "Putting it all together".
Running it locally requires pip installing tensorflow as well as some other libraries such as opencv, which are already preinstalled on google colab
as well as explicitly importing other libraries. Upload the zips files to google colab as well. Basically, the end product of the notebook 
from export inference graph will produce a new frozen_inference_graph.pb in your new_checkpoint directory that you will then use as a standalone
file for production.

## Production

### App:
Final app is hosted on GKE (Google Kubernetes Engine) with the frontend utilizing Streamlit, whereby a user can pass in a waldo puzzle jpeg
and the app will return if it can detect waldo in the image and the coordinates of Waldo in the image as well.

Link to app - http://146.148.91.238/

### Description of files:
1. deployment.yaml - deployment config for app on kubernetes
2. service.yaml - service config to expose app to internet on kubernetes
3. Dockerfile - file to create docker image, basis for kubernetes pod
4. requirements.txt - python packages that are called in the Dockerfile
5. labels.txt - label for waldo
6. frozen_inference_graph.pb - new graph produced at end of wheres_waldo notebook
7. waldo_detector.py - app logic that includes 

