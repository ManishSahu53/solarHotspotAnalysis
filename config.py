IMAGE_SIZE = 416
CLASSES = 'model_data/friends_classes.txt'
WEIGHTS = 'model_data/yolo_weights.h5'
ANCHORS = 'model_data/yolo_anchors.txt'
ANNOTATATIONS = 'model_data/annotation.txt'
LOGS = 'logs'
TINY_WEIGHTS = 'model_data/tiny_yolo_weights.h5'


# Parameters
SCORE = 0.4
IOU = 0.65
NUM_GPU = 1
EPOCH = 10

# Mapping
MAPPING = {
    'joey': 0,
    'chandler':1,
    'ross':2,
    'monica':3,
    'rechal':4,
    'pheobe':5
}


FORMAT = ['.JPEG', '.JPG', '.PNG']