import keras
import sys, os
from glob import glob
import progressbar
import imageio
import argparse
import logging
from ast import literal_eval

import os, sys, os.path as op
sys.path.append(op.join(os.getenv('SHUFFLER_DIR'), 'interface'))
from interfaceKeras import BareImageGenerator  # To grab images.
from shufflerDataset import DatasetWriter  # To write detected bboxes into a database.

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# import keras_retinanet
from .. import models
from ..utils.image import read_image_bgr, preprocess_image, resize_image
from ..utils.visualization import draw_box, draw_caption
from ..utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

parser = argparse.ArgumentParser(description='Detect boxes in images from Shuffler database')
parser.add_argument('-i', '--in_db_file',
        help='Path to Shuffler database that contains paths to test images.',
        default='../stamps-1800x1200-empty-1class.db')
parser.add_argument('-o', '--out_db_file',
        help='The path to a new Shuffler database, where detections will be stored.',
        default='examples/detected/epoch10-test.db')
parser.add_argument('--rootdir',
        help='Where image files in the db are relative to.',
        default='..')
parser.add_argument('--coco_category_id_to_name_map', default='{0: "stamp"}',
        help='A map (as a json string) from category id to its name.')
parser.add_argument('--out_dir',
        help='The path of the output directory. ',
        default='examples/detected/epoch10-test')
parser.add_argument('--model_path',
        help='The path of the trained model .h5 file.',
        default='snapshots/from-coco-weights/resnet50_coco_10.h5')
parser.add_argument('--logging', default=20, type=int, choices={10, 20, 30, 40},
        help='Log debug (10), info (20), warning (30), error (40).')
args = parser.parse_args()

progressbar.streams.wrap_stderr()
progressbar.streams.wrap_stdout()
FORMAT = '[%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s'
logging.basicConfig(level=args.logging, format=FORMAT)


# Get data generator.
generator = BareImageGenerator(
        db_file=args.in_db_file,
        rootdir=args.rootdir,
        batch_size=32,
        shuffle=False)


# Get shuffler dataset writer
datasetWriter = DatasetWriter(out_db_file=args.out_db_file,
        rootdir=args.rootdir,
        overwrite=True)


# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

print ('Loading model...')
model = models.load_model(args.model_path, backbone_name='resnet50')
model = models.convert_model(model)

labels_to_names = literal_eval(args.coco_category_id_to_name_map)

videowriter = imageio.get_writer(args.out_dir + '.mp4', fps=1, codec='mjpeg')

for batch in progressbar.progressbar(generator):
  images = batch['image']
  imagefiles = batch['imagefile']
  draws = []
  scales = []

  print ('Preprocessing batch... ', end='')
  start = time.time()
  for i, image in enumerate(images):

    draw = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    images[i] = image
    draws.append(draw)
    scales.append(scale)

  images = np.stack(images)
  #print (images.shape)

  print("\tin %.2f sec." % (time.time() - start))

  # process images
  print ('Predicting batch... ', end='')
  start = time.time()
  boxess, scoress, labelss = model.predict_on_batch(images)
  print("\tin %.2f sec." % (time.time() - start))

  print ('Postprocessing batch.. ', end='')
  for image, imagefile, draw, scale, boxes, scores, labels in zip(images, imagefiles, draws, scales, boxess, scoress, labelss):

    datasetWriter.addImage({'imagefile': imagefile})
    
    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score in zip(boxes, scores):
        label = 0
        # scores are sorted so we can break
        if score < 0.1:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "%.3f" % score
        #caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

        datasetWriter.addObject({'imagefile': imagefile, 
            'x1': int(b[0]), 'y1': int(b[1]), 'width': int(box[2]-box[0]), 'height': int(box[3]-box[1]),
            'name': labels_to_names[label],
            'score': float(score)})
        
    out_image_path = os.path.join(args.out_dir, os.path.basename(imagefile))
    imageio.imwrite(out_image_path, draw)
    videowriter.append_data(draw)

  print("\tin %.2f sec." % (time.time() - start))

videowriter.close()
datasetWriter.close()

