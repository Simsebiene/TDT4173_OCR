#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

def sliding_window(image, window_size, stride):
    """
    Takes an input image, then slices off predictions of size $window_size.
    Window moves according to $stride.
    :param image: input image as PIL image object
    :param window_size: size of the sliding window, tuple of integers (x,y)
    :param stride: step size of the sliding window, tuple of integers (x,y)
    :yields: tuple (x_min <int>, y_min <int>, image slice <PIL image>)
    """
    for x in range(0, image.size[0]-window_size[0], stride[0]):
        for y in range(0, image.size[1]-window_size[1], stride[1]):
            yield (x, y, image.crop((x,y,x+window_size[0],y+window_size[1])))

       
# ToDo:
# The predict() function should be replaced by an actual classifier.
# You can reuse it for feature-engineering, if you want to!
# The new classifier should return a class prediction and corresponding confidence score for a given image slice.
# 26 + 1 classes (one class for "no character found!")
import string
import random 
def predict(crop, num_pixels):
    """
    Dummy predictor!
    Returns the input image, the prediction and confidence score.
    Just random classes right now. Chooses class "0" if all pixels in the window are white.
    Confidence is the ratio of nonwhite pixels (< 255) to total pixels.
    :param crop: tuple (x_min <int>, y_min <int>, image slice <PIL image>), a slice of the input image
    :param num_pixels: integer, size of the box in pixels
    :returns: predicted class as string, confidence score as float
    """
    background_color = 255
    prediction = "0" if np.min(crop[2])>=background_color else random.choice(string.ascii_lowercase)
    mask = (np.array(crop[2]) < background_color)
    confidence = np.array(crop[2])[mask].size / 400.  
    return prediction, confidence


def scan(image, window_size, stride, score_threshold):
    """
    Returns a list of predictions, with their predicted letters and positions.
    :param image: input image as PIL image object
    :param window_size: size of the sliding window, tuple of integers (x,y)
    :param stride: step size of the sliding window, tuple of integers (x,y)
    :param score_threshold: float, minimum confidence score to be considered a hit
    :returns: 
    """
    num_pixels = window_size[0]*window_size[1]
    boxes = []
    for crop in sliding_window(image, window_size, stride):
        prediction, confidence = predict(crop,num_pixels)
        if confidence >= score_threshold and prediction != "0":
            boxes.append((crop[0], crop[1], prediction, confidence))
    return boxes


def get_iou(box1, box2):
    """
    Calculate the intersection over union (IOU) between box1 and box2.
    :param box1: numpy array with shape [x_min,y_min,x_max,y_max]
    :param box2: numpy array with shape [x_min,y_min,x_max,y_max]
    :returns: "intersection over union" as float
    """
    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its area.
    x_min = max(box1[0],box2[0])
    y_min = max(box1[1],box2[1])
    x_max = min(box1[2],box2[2])
    y_max = min(box1[3],box2[3])
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    inter = float((x_max-x_min)*(y_max-y_min)) if x_max > x_min and y_max > y_min else 0
    union = float((box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter)
    # compute the IoU
    iou = inter/union
    return iou


def nms(boxes, window_size, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies non-max suppression (NMS) to a set of boxes. 
    :param boxes: list of tuples (x_min <int>, y_min <int>, class prediction <string>, confidence score <float>)
    :param max_boxes: integer, maximum allowed detections
    :param iou_threshold: integer, "intersection over union" threshold used for NMS filtering
    :returns: NMS filtered list of tuples (x_min <int>, y_min <int>, class prediction <string>, confidence score <float>)
    """
    # Break list of tuples down into numpy arrays. (Ugly, but the easiest way) 
    classes = np.array([box[2] for box in boxes])
    scores = np.array([box[3] for box in boxes])
    boxes_coords = np.array([[box[0],box[1],box[0]+window_size[0],box[1]+window_size[1]] for box in boxes])
    nms_indices = []
    # Use get_iou() to get the list of indices corresponding to boxes you keep
    idxs = np.argsort(scores)
    while len(idxs) > 0 and len(nms_indices) < max_boxes:
        last = len(idxs) - 1
        ind_max = idxs[last]
        nms_indices.append(ind_max)
        suppress = [last]
        for i in range(0,last):
            overlap = get_iou(boxes_coords[ind_max], boxes_coords[idxs[i]])
            if overlap > iou_threshold:
                suppress.append(i)
        idxs = np.delete(idxs, suppress)
    # Use index arrays to select only nms_indices from boxes, scores, and classes
    boxes = [(boxes_coords[index,0], boxes_coords[index,1], 
              classes[index], scores[index]) for index in nms_indices]
    return boxes


def plot(image, boxes, window_size):
    """
    Plots the bounding boxes with class label and confidence score on top of the original image.
    :param image: input image as PIL image object 
    :param boxes: list of tuples (x_min <int>, y_min <int>, class prediction <string>, confidence score <float>)
    """
    fig1 = plt.figure(dpi=200)
    ax1 = fig1.add_subplot(1,1,1) 
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.axis('off')
    
    for b, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box[0]-.5, box[1]-.5, box[0]+window_size[0]-.5, box[1]+window_size[1]-.5
        prediction, score = box[2], box[3]
        ax1.text(x_min, y_min-3, "%s %d%%" % (prediction, score*100), color="red", fontsize=3)
        x = [x_max, x_max, x_min, x_min, x_max]
        y = [y_max, y_min, y_min, y_max, y_max]
        line, = ax1.plot(x,y,color="red")
        line.set_linewidth(.5)
    return


def main():
    # Load the image
    image = Image.open("./dataset/detection-images/detection-2.jpg")
    # Hyperparameters
    window_size         = (20,20)
    stride              = (1,1)
    score_threshold     = .85 # Minimum confidence score to be considered a hit.
    iou_threshold       = .1 # Too high --> false positives occur.
    max_boxes           = 100
    # Run everything
    boxes = scan(image, window_size, stride, score_threshold)
    boxes = nms(boxes, window_size, max_boxes, iou_threshold)
    plot(image, boxes, window_size)


if __name__== "__main__":
    main()
