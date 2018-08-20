"""
Usage:
  # From command prompts
  # Create train data:
  python generate_tfrecord.py --image_input=E:/healthy/  --output_path=C:/tensorflow1/models/research/object_detection/train.record --folder_name=train

  # Create test data:
  python generate_tfrecord.py --image_input=E:/healthy/  --output_path=C:/tensorflow1/models/research/object_detection/train.record --folder_name=train
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

import glob
import xml.etree.ElementTree as ET

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('image_input', '', 'Path to the Image files')
flags.DEFINE_string('output_path', '', 'Path to output csv and TFRecord')
flags.DEFINE_string('folder_name', '', 'Category of train or test')
FLAGS = flags.FLAGS

# change this with your custom label
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'healthy':
        return 1
    elif row_label == 'scorched_plant':
        return 2
    elif row_label == 'worm_eaten_plant':
        return 3
    elif row_label == 'scorched_part':
        return 4
    elif row_label == 'eaten_hole':
        return 5
    else:
        None
        
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    
    image_path = str(FLAGS.image_input)
    xml_df = xml_to_csv(image_path)
    print(str(FLAGS.output_path))
    output_path=os.path.dirname(os.path.realpath(str(FLAGS.output_path)))
    csv_location = (str(output_path) +"/"+ str(FLAGS.folder_name)+ '_labels.csv')
    xml_df.to_csv(csv_location, index=None)
    print('Successfully converted xml to csv.')
    
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    print(FLAGS.output_path)

    path = image_path
    #os.path.join(os.getcwd(), 'images')
    examples = pd.read_csv(csv_location)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
