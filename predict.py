##############################
#Predict
import sys
import os

#args = sys.argv

#args[0] is the name of this script
#ROOT_DIR = args[1] 
#model_path = args[2]
#DATA_DIR = args[3]
#subset = args[4]

#ROOD_DIR = "H:\\SpeciesClassification\\arundo\\code\\Mask_RCNN-upgraded"
ROOT_DIR = "H:\\SpeciesClassification\\arundo\\Mask_RCNN-Lee"
model_path = os.path.join(ROOT_DIR, "mask_rcnn_arundo_0027.h5")
#DATA_DIR = "H:\\SpeciesClassification\\arundo\\cocodataset_mixed\\dataset"
DATA_DIR = "H:\\SpeciesClassification\\arundo\\cocodataset3_splittile1024\\dataset"
#DATA_DIR = "H:\\SpeciesClassification\\arundo\\cocodataset2_splittile2048_to_1024\\dataset"
subset = "train"

DETE_DIR = os.path.join(os.path.join(ROOT_DIR, "detections"), "exp")
IMG_DIR = os.path.join(DETE_DIR, "images")
SHP_DIR = os.path.join(DETE_DIR, "shapes")

if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)

if not os.path.exists(SHP_DIR):
    os.mkdir(SHP_DIR)

imagepath = os.path.join(IMG_DIR, subset)
if not os.path.exists(imagepath):
    os.mkdir(imagepath)

sys.path.append(ROOT_DIR)
sys.path.append(DATA_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

from DetectorConfig import DetectorConfig 
from DetectorDataset import DetectorDataset

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mask2geotif import convert_mask

inference_config = DetectorConfig()

ORIG_SIZE = 1024

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir=ROOT_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors

# prepare the validation dataset
dataset = DetectorDataset(ORIG_SIZE, ORIG_SIZE, DATA_DIR)
dataset.load_custom(DATA_DIR, subset = subset)
dataset.prepare()

for image_id in dataset.image_ids:
#for i in range(8):
    #image_id = random.choice(dataset.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, inference_config, 
                               image_id, augmentation=None)
    
    fig = plt.figure(figsize=(20, 10))
    #plt.subplot(num, 2, 2*i + 1)
    plt.subplot(1, 2, 1)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset.class_names,
                                colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
    
    #plt.subplot(num, 2, 2*i + 2)
    image_path = dataset.image_info[image_id]['path']
    maskfilename = os.path.split(image_path)[1]
    savepath = os.path.join(imagepath, maskfilename)
    
    plt.subplot(1, 2, 2)
    results = model.detect([original_image]) #, verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], 
                                colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
    if np.sum(r['masks'], axis = None)!=0:
        convert_mask(image_path,
                     r['masks'],
                     r['class_ids'],
                     dataset.class_names,
                     r['scores'],
                     savepath)
    #plt.show()
    #info = dataset.image_info[image_id]
    #path = info['path']
#    filename = dataset.image_names[i]
#    filepath = os.path.join(os.path.join(ROOT_DIR, 'mask'), filename)
#    convert_mask(original_image, r['masks'], r['class_ids'], dataset.class_names, r['scores'], filepath)
    

    # Save just the portion _inside_ the second axis's boundaries
    #ax2 = fig.axes[i][:1]
    #extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #fig.savefig(os.path.join(DATA_DIR,'val_'+str(i)+'.png'), dpi=100, bbox_inches=extent)
    #i += 1
    name = dataset.image_info[image_id]['path']
    name = os.path.basename(name)
    name = name.replace('.tif','_gt_det.png')
    fig.savefig(os.path.join(DATA_DIR,name), dpi=100)
    
#import gdal
#import glob
#for file in glob.glob("dir/*.asc"):
#    new_name = file[:-4] + ".json"
#    gdal.Polygonize(file, "-f", "GeoJSON", new_name)

#sudo apt-get install python-gdal
#convert each mask.tif to mask.shp
#gdal_polygonize.py mask1.tif -f "ESRI Shapefile" mask1.shp
#merge shapefiles together
#ogr2ogr -f "ESRI Shapefile" -update -append merge.shp part1.shp -nln merge
