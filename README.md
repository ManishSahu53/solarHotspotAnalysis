# Introduction
This repo is used to get automatic hotspot detection and analysis from thermal images taken from DJI's zenmuse XT-2 drone flown above a solar plant.
The height of drone was kept 30m from the ground. This complete task is divided into following parts:

* Hotspot Detection

    Hotspot detection is done using yolov3 object detection technique. Detialed explaination about the method can be found [here](https://github.com/qqwweee/keras-yolo3).

* Hotspot Analysis 

    This includes area of damage, temperature of the damage. Since each image is in RJPG (Radiometric JPEG) format, we can extract tempearture values from the brightness value of the image.

* Locating damage via image's geo-coordinate 

## Tutorial
To successfully run this repo, do the following steps:

* [convert.py](https://github.com/ManishSahu53/solarHotspotAnalysis/convert.py): First download yolov3 weights and convert it to hdf5 format.


```
    wget https://pjreddie.com/media/files/yolov3.weights
    python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5

```
* [generateAnnotation.py](https://github.com/ManishSahu53/solarHotspotAnalysis/generateAnnotation.py) : 
This is used to generate bbox data format in yolo format. I have used [labelme](https://github.com/wkentaro/labelme)
to draw bounding boxes. Keep the images and json to `data/train` and then run following code:

```
    python generateAnnotation.py --data data/train/ --output solarAnnotation.txt

```

* [trainObjectDetection.py](https://github.com/ManishSahu53/solarHotspotAnalysis/trainObjectDetection.py):
This is used to run yolo model. There are multiple parameters inputs to the repo.

```
    data : Input file containing annotation path and image path, this is generated from generateAnnotatio.py
    classes : Input file containing classes
    anchor : Input file containing anchors
    size : Input file size. [Default] - 416x416
    epoch : Enter number of epochs for training. [Default] 10
    batch : Enter number of batch size for training. [Default] 4

    python trainObjectDetection.py --data solarAnnotation.txt --classes objectDetection/model_data/thermal_classes.txt --anchor objectDetection/model_data/yolo_anchors.txt --size 416 --epoch 10 --batch 4 --output 
```

## YoloV3
YoloV3 code has been taken frtom [Here](https://github.com/qqwweee/keras-yolo3)
