import hashlib
import io
import os
import pandas as pd
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('annotations_path', '', 'Path to annotiations.csv.')
flags.DEFINE_string('images_directory', '', 'Path to images directory.')
FLAGS = flags.FLAGS

def combine_annotations_and_image(row, image_directory):

  """
  Args:
    row: row of annotations from pandas iterrows
    image_directory: String specifying directory containing images
  """
  img_path = os.path.join(image_directory, row['filename'])
  with tf.gfile.GFile(img_path,'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  height=int(row['height'])
  width=int(row['width'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []

  # normalized coordinates
  xmin.append(float(row['xmin']) / width)
  ymin.append(float(row['ymin']) / height)
  xmax.append(float(row['xmax']) / width)
  ymax.append(float(row['ymax']) / height)
  classes_text.append(row['class'].encode('utf8'))
  classes.append(1)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(row['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(row['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes)
  }))
  return example

def create_tf_record(output_filename,
                     images_dir,
                     examples):
  """Creates a TFRecord file from examples.
  Args:
    output_filename: Path to where output file is saved.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)

  for example in examples.iterrows():

    tf_example = combine_annotations_and_image(example[1], images_dir)

    writer.write(tf_example.SerializeToString())

  writer.close()


# TODO: Add test for pet/PASCAL main files.
def main(_):

  # split 70/30 for train/test with the annotations.csv
  annotations_df=pd.read_csv(FLAGS.annotations_path)
  train_examples=annotations_df.sample(frac=0.7,random_state=42) #random state is a seed value
  val_examples=annotations_df.drop(train_examples.index)
  
  print(f"training - {list(train_examples['filename'])},\n test - {list(val_examples['filename'])}")

  train_output_path = 'train.tfrecord'
  val_output_path = 'test.tfrecord'

  create_tf_record(train_output_path, FLAGS.images_directory,train_examples)
  create_tf_record(val_output_path, FLAGS.images_directory,val_examples)

if __name__ == '__main__':
  tf.app.run()