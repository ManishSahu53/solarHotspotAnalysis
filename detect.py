from __future__ import division

from models import *
from utils.util import *
from utils.datasets import *
import json

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from thermal import getTemperature


def load_model(model_def, img_size, weights_path, device):
    #  Set up model
    model = Darknet(model_def, img_size=img_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()  # Set in evaluation mode
    return model


def post_process_prediction(mapping, img, img_size, path, detections, classes, is_temperature=False):
    # Bounding-box colors
    # print(f'############# image shape: {img.shape}')

    cmap = plt.get_cmap("brg")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Reading temperatures
    if is_temperature is True:
        temperature = getTemperature.getTemp(path)

    # Draw bounding boxes and labels of detections
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)

        # Drawing all the Boudning Bbbox
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            temp = {
                # 'class': str(float(cls_pred)),
                'class': 'HotSpot',
                'class_confidence': str(float(cls_conf)),
                'bbox': [str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2))],
            }
            if is_temperature is True:
                temp['temperature_stats(Celcius)'] = getTemperature.getStas(
                    temperature, [int(x1), int(y1), int(x2), int(y2)])

            mapping[path].append(temp)
            print("\t+ Label: %s, Conf: %.5f" %
                  (classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(
                np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle(
                (x1, y1), box_w*1.2, box_h*1, .2, linewidth=1, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s='HotSpot',
                color="Red",
                verticalalignment="bottom",
                bbox={"color": color, "pad": 0},
            )

    return plt, fig, mapping


def process(image_folder, model_def, weights_path,
            class_path, conf_thres, nms_thres, batch_size,
            n_cpu, img_size, device, path_output):

    model = load_model(model_def, img_size, weights_path, device=device)

    dataloader = DataLoader(
        ImageFolder(image_folder, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
    )

    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # print(f'############# image shape: {input_imgs.shape}')
        # Get detections    
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    print("\nSaving images:")
    mapping = {}

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        mapping[path] = []

        # Create plot
        img = np.array(Image.open(path))
        plt, fig, mapping = post_process_prediction(
            mapping, img, img_size, path, detections, classes, is_temperature=is_temperature)

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f'{path_output}/{filename}.png',
                    bbox_inches="tight", pad_inches=0.0)
        plt.close()

    with open(f'{path_output}/result.json', 'w') as outfile:
        json.dump(mapping, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str,
                        default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str,
                        default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str,
                        default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float,
                        default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2,
                        help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int,
                        default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416,
                        help="size of each image dimension")
    parser.add_argument("--path_output", default=416,
                        type=str, help="otuput of the prediction")

    opt = parser.parse_args()
    print(opt)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except:
        device = 'cpu'

    image_folder = opt.image_folder
    model_def = opt.model_def
    weights_path = opt.weights_path
    class_path = opt.class_path
    conf_thres = opt.conf_thres
    nms_thres = opt.nms_thres
    batch_size = opt.batch_size
    n_cpu = opt.n_cpu
    img_size = opt.img_size
    path_output = opt.path_output
    os.makedirs("path_output", exist_ok=True)

    is_temperature = True
    process(image_folder=image_folder, model_def=model_def, weights_path=weights_path,
            class_path=class_path, conf_thres=conf_thres, nms_thres=nms_thres, batch_size=batch_size,
            n_cpu=n_cpu, img_size=img_size, device=device,
            path_output=path_output)
