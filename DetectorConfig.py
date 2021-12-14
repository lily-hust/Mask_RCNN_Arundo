# The following parameters have been selected to reduce running time for demonstration purposes 
# These are not optimal 
import os
ROOT_DIR = r'H:\SpeciesClassification\arundo\Mask_RCNN-Lee'
os.chdir(ROOT_DIR)
from mrcnn.config import Config

debug = False
class DetectorConfig(Config):    
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name  
    NAME = 'arundo'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False
    
    BACKBONE = 'resnet101'
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    
    # Number of classes (including background)
    NUM_CLASSES = 2  
    
    #ship searching example use 384-384, bigger than pneumonia finding example of 256-256, because the former example's targets are more than the latter and distribute everywhere
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    TRAIN_ROIS_PER_IMAGE = 64
    
    MAX_GT_INSTANCES = 14
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.95
    DETECTION_NMS_THRESHOLD = 0.0

    # Use a small epoch since the data is simple
    #STEPS_PER_EPOCH = 190
    #VALIDATION_STEPS = 28
    STEPS_PER_EPOCH = 10 if debug else 50
    VALIDATION_STEPS = 5 if debug else 5
    
    ## balance out losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 30.0,
        "rpn_bbox_loss": 0.8,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.2
    }

       
