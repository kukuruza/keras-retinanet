import keras
import sys, os
from glob import glob
import progressbar
import imageio
import argparse

import os, sys, os.path as op
sys.path.append(op.join(os.getenv('SHUFFLER_DIR'), 'interface'))
from interfaceKeras import BareImageGenerator

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

progressbar.streams.wrap_stdout()

parser = argparse.ArgumentParser(description='Detect boxes in images from Shuffler database')
parser.add_argument('-i', '--in_db_file',
        default='../stamps-1800x1200-empty-1class.db')
parser.add_argument('--rootdir',
        help='Where image files in the db are relative to.',
        default='..')
parser.add_argument('--out_dir',
        help='The path of the output directory. ',
        default='examples/detected/epoch10-test')
parser.add_argument('--model_path',
        help='The path of the trained model .h5 file.',
        default='snapshots/from-coco-weights/resnet50_coco_10.h5')
args = parser.parse_args()


# Get data generator.
generator = BareImageGenerator(
        db_file=args.in_db_file,
        rootdir=args.rootdir,
        batch_size=32,
        shuffle=False)


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

#labels_to_names = {0: 'stamp'}

writer = imageio.get_writer(args.out_dir + '.mp4', fps=1, codec='mjpeg')

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

    # correct for image scale
    boxes /= scale
#    print (boxes)
#    print (scores.max())

    # visualize detections
    for box, score in zip(boxes, scores):
        label = 0
        # scores are sorted so we can break
        if score < 0.5:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "%.3f" % score
        #caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        
    out_image_path = os.path.join(args.out_dir, os.path.basename(imagefile))
    imageio.imwrite(out_image_path, draw)
    writer.append_data(draw)

  print("\tin %.2f sec." % (time.time() - start))

writer.close()

