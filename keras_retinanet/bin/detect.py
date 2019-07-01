import keras
import sys, os
from glob import glob
from progressbar import progressbar
import imageio

#import os, sys, os.path as op
#sys.path.append(op.join(os.getenv('SHUFFLER_DIR'), 'interface'))
#from kerasInterface import BareImageGenerator

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

# Get data generator.
#generator = BareImageGenerator(
#        db_file='labelme/Images/HistoricalDocuments-1800x1200/258-00-01.jpg',
#        rootdir='/home/etoropov/projects/HistoricalDocuments',
#        batch_size=32,
#        shuffle=False)

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  return tf.Session(config=config)

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

print ('Loading model...')
model_path = os.path.join('snapshots/from-coco-weights/resnet50_coco_10.h5')
model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

labels_to_names = {0: 'stamp'}

image_pattern = '../datasets/stamps-1800x1200-1class-empty/*.jpg'
image_paths = glob(image_pattern)

writer = imageio.get_writer('examples/detected/epoch10-test.avi', fps=1)

for image_path in progressbar(image_paths):

    print ('Preprocessing image...')
    image = read_image_bgr(image_path)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    print ('Predicting image...')
    start = time.time()
    output = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes, scores, labels = output
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    print (boxes)
    print (scores.max())

    # visualize detections
    for box, score in zip(boxes[0], scores[0]):
        label = 0
        # scores are sorted so we can break
        if score < 0.5:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        
    out_image_path = os.path.join('examples/detected/epoch10-test', os.path.basename(image_path))
    imageio.imwrite(out_image_path, draw)
    writer.append_data(draw)

writer.close()

