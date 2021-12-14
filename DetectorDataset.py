import os
import json
import numpy as np
# this is too slow for showing warning of 'low contrast image'
from skimage.io import imread
#from PIL import Image

ROOT_DIR = r'H:\SpeciesClassification\arundo\Mask_RCNN-Lee'
os.chdir(ROOT_DIR)
from mrcnn import utils

class DetectorDataset(utils.Dataset):
    """Dataset class for training our dataset.
    """
        #image_ids refer to a np.array saving the image IDs
        #annotation file name's naming rule is to cat(image_name.split('.')[0], '_greenhouse_', corresponding_annotation_id)
      
    def __init__(self, ORIG_HEIGHT, ORIG_WIDTH, DATA_DIR):
        super().__init__(self)
        self.ORIG_HEIGHT = ORIG_HEIGHT
        self.ORIG_WIDTH = ORIG_WIDTH
        self.DATA_DIR = DATA_DIR
        # Add classes
        #self.add_class('greenhouse', 1, 'greenhouse')
      
    def load_custom(self, dataset_dir, subset):
        #ORIG_SIZE = 1024
        # dataset_dir = DATA_DIR
        """Load a subset of the bottle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("arundo", 1, "arundo")
        #self.add_class("object", 2, "balloon")
 
        # Train or validation dataset?
        assert subset in ["train", "eval", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        image_dir = os.path.join(dataset_dir, "images")
        annotation_dir = os.path.join(dataset_dir, "annotations")

        # find image_names, image_ids, annotations from json file
        annotations = json.load(open(os.path.join(dataset_dir, "instances_arundo_" + subset + ".json")))
        image_ids = [a['id'] for a in annotations['images']]
        image_names = [a['file_name'] for a in annotations['images']]

        # add images 
        for i, image_name in enumerate(image_names):
            # get corresponding annotation file names for this image name
            path = os.path.join(image_dir, image_name)
            #pre = image_name.split('.')[0]+'_greenhouse_'
            #image_annotations = [f for f in os.listdir(annotation_dir) if f.startswith(pre)]
            self.add_image('arundo', image_ids[i], path, 
                           orig_height = self.ORIG_HEIGHT, orig_width = self.ORIG_WIDTH)
            
        
            
    def image_reference(self, image_id):
      #image_id may not be the self.image_info order. so here image_id refers to the order in self.image_info, starting from 0
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
      #here image_id refers to the order in self.image_info, starting from 0
        info = self.image_info[image_id]
        path = info['path']
        image = imread(path)
        #image = Image.open(path)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    # json records polygon instead of target pixels, so can't use rle things convertion functions 
    def load_mask(self, image_id):
      # from self.image_info[image_id]['path'] can get the full path of the image. Then extract the folder part from the path, and then extract "train" or "eval", "test" and construct the path for the masks
        # need to get image_annotations, categories_of_annotations from image_id
        # image_annotations refers to the annotation paths corresponding to a certain image_info.id

        info = self.image_info[image_id]
        id = info['id']

        image_path = info['path']
        image_folder = os.path.split(image_path)[0]
        subset = os.path.split(os.path.split(image_folder)[0])[1]

        dataset_dir = os.path.join(self.DATA_DIR, subset)
        annotation_dir = os.path.join(dataset_dir, 'annotations')

        annotations = json.load(open(os.path.join(dataset_dir, "instances_arundo_" + subset + ".json")))
        image_annotation_segs = [a['segmentation'] for a in annotations['annotations'] if a['image_id'] == id]
        
        pre = os.path.split(image_path)[1].split('.')[0]+'_'
        image_annotation_files = [f for f in os.listdir(annotation_dir) if f.startswith(pre)]

        count = len(image_annotation_files)
        print(image_path, count)

        if count == 0:
            mask = np.zeros((self.ORIG_HEIGHT, self.ORIG_WIDTH, 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((self.ORIG_HEIGHT, self.ORIG_WIDTH, count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(image_annotation_files):
                mask[:,:,i] = imread(os.path.join(annotation_dir,a))
                #mask[:,:,i] = Image.open(os.path.join(annotation_dir,a))
        #        # class_ids[i] should be categories_of_annotations[i]
                class_ids[i] = 1 
        return mask.astype(np.bool), class_ids.astype(np.int32)   


    def load_display_mask(self, image_id, subset):
        # need to get image_annotations, categories_of_annotations from image_id
        # image_annotations refers to the annotation paths corresponding to a certain image_info.id

        _dir = os.path.join(self.DATA_DIR, subset)

        info = self.image_info[image_id]
        id = info['id']
        annotations = json.load(open(os.path.join(_dir, "instances_arundo_" + subset + ".json")))
        image_annotation_segs = [a['segmentation'] for a in annotations['annotations'] if a['image_id'] == id]
        
        count = len(image_annotation_segs)
        if count == 0:
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(image_annotation_segs):
                class_ids[i] = 1 
        return  image_annotation_segs, class_ids.astype(np.int32)
