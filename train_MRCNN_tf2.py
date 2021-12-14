#pip install keras==2.2.5
#succeed
#pip install h5py==2.10

#### tensorflow by default now is 2.6 , need to downgrade to 1.x to run mask rcnn
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

import os 
import sys
from imgaug import augmenters as iaa
import pandas as pd 
import glob

ROOT_DIR = r'H:\SpeciesClassification\arundo\Mask_RCNN-Lee'
os.chdir(ROOT_DIR) #this does not work when on anaconda prompt call python *.py

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.config import Config
# Import COCO config

from DetectorConfig import DetectorConfig 
from DetectorDataset import DetectorDataset

DATA_DIR = r'H:\SpeciesClassification\arundo\reinforce_2\dataset'
#DATA_DIR = r'H:\SpeciesClassification\arundo\cocodataset2_splittile2048_to_1024\dataset'
#DATA_DIR = r'H:\SpeciesClassification\arundo\cocodataset_merged\dataset'
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_arundo_0020.h5") 1st stage best epoch, train heads
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_arundo_0027.h5") #2nd stage best epoch, train all
ORIG_SIZE = 1024

#%%time
# prepare the training dataset

dataset_train = DetectorDataset(ORIG_SIZE, ORIG_SIZE, DATA_DIR)
dataset_train.load_custom(dataset_dir = DATA_DIR, subset = "train")
dataset_train.prepare()

# prepare the validation dataset
dataset_val = DetectorDataset(ORIG_SIZE, ORIG_SIZE, DATA_DIR)
dataset_val.load_custom(dataset_dir = DATA_DIR, subset = "eval")
dataset_val.prepare()

# Image augmentation (light but constant)
augmentation = iaa.Sequential([
    iaa.OneOf([ ## rotate
        iaa.Affine(rotate=0),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270),
    ]),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])

config = DetectorConfig()
#config.display()
model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

#%%time
LEARNING_RATE = 0.00001

# Train Mask-RCNN Model 
import warnings 
warnings.filterwarnings("ignore")

## train heads with higher lr to speedup the learning
#COCO_WEIGHTS_PATH = r'H:\SpeciesClassification\arundo\result\size1024\mask_rcnn_arundo_0040.h5'
#model.load_weights(COCO_WEIGHTS_PATH, by_name=True)

#my_callbacks = [
#    tf.keras.callbacks.EarlyStopping(patience=2),
#    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
#    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
#]
#model_cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ROOT_DIR, 'detections'))
#history_cb = tf.keras.callbacks.CSVLogger('./log.csv', separator=",", append=False)
model_inference = modellib.MaskRCNN(mode='inference', config=config, model_dir=ROOT_DIR)
mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model,
                                                                        model_inference,
                                                                        dataset_val,
                                                                        calculate_map_at_every_X_epoch=10,
                                                                        verbose=1)
EPOCHS = 100
model.train(dataset_train,
            dataset_val,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            layers='heads',
            augmentation=augmentation,
            custom_callbacks=[mean_average_precision_callback])

#model.train(dataset_train, dataset_val,
#            learning_rate=LEARNING_RATE,
#            epochs=5,
#            layers='heads',
#            augmentation=augmentation)
##            custom_callbacks=[model_cb, history_cb])  ## no need to augment yet

history = model.keras_model.history.history
# saving the history of the model. using json.dump does not work any more in tensorflow.keras
#import json
#with open('trainHistory.json', 'w') as handle: 
#    json.dump(history, handle)
np.save('my_history.npy',history)

with open("./result_model.txt",'w') as f:
    for k in history.keys():
        print(k,file=f)
        for i in history[k]:
            print(i,file=f)

import numpy as np
best_epoch = np.argmin(history["val_loss"])
score = history["val_loss"][best_epoch]

train_loss = history["loss"][best_epoch]
file1 = open("bestepoch.txt","w")
best_epoch += 1
L = 'Best epoch is ' + str(best_epoch) + ' val_loss is ' +str(score) + ' train_loss is ' + str(train_loss)
file1.write(L)
file1.close()

fig = plt.figure(figsize=(21,11))
epochs = [i for i in range(EPOCHS)]
plt.subplot(231)
plt.plot(epochs, history["loss"], label="Train loss")
plt.plot(epochs, history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(232)
plt.plot(epochs, history["rpn_class_loss"], label="Train RPN class ce")
plt.plot(epochs, history["val_rpn_class_loss"], label="Valid RPN class ce")
plt.legend()
plt.subplot(233)
plt.plot(epochs, history["rpn_bbox_loss"], label="Train RPN box loss")
plt.plot(epochs, history["val_rpn_bbox_loss"], label="Valid RPN box loss")
plt.legend()
plt.subplot(234)
plt.plot(epochs, history["mrcnn_class_loss"], label="Train MRCNN class ce")
plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid MRCNN class ce")
plt.legend()
plt.subplot(235)
plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train MRCNN box loss")
plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid MRCNN box loss")
plt.legend()
plt.subplot(236)
plt.plot(epochs, history["mrcnn_mask_loss"], label="Train Mask loss")
plt.plot(epochs, history["val_mrcnn_mask_loss"], label="Valid Mask loss")
plt.legend()

#plt.show()

name = 'training_190_'+str(EPOCHS)+'.png'
fig.savefig(name, dpi=300)
#fig.savefig('training_190_{EPOCHS}.png', dpi = 300)


