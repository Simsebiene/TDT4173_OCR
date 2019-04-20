#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


def detect(crop, num_pixels):
    """
    Detector. Returns detection score for a given image slice.
    Detection score is the ratio of nonwhite pixels (< 255) to total pixels.
    :param crop: tuple (x_min <int>, y_min <int>, image slice <PIL image>), a slice of the input image
    :param num_pixels: integer, size of the box in pixels
    :returns: detection score as float
    """
    background_color = 255
    mask = (np.array(crop) < background_color)
    detect_score = np.array(crop)[mask].size / 400.  
    return detect_score


def scan(image, window_size, stride, detect_score_threshold):
    """
    Returns a list of predictions, with their predicted letters and positions.
    :param image: input image as PIL image object
    :param window_size: size of the sliding window, tuple of integers (x,y)
    :param stride: step size of the sliding window, tuple of integers (x,y)
    :param score_threshold: float, minimum detection score to be considered a hit
    :returns: list of tuples (x_min <int>, y_min <int>, detection score <float>)
    """
    num_pixels = window_size[0]*window_size[1]
    boxes = []
    for roi in sliding_window(image, window_size, stride):
        detect_score = detect(roi[2],num_pixels)
        if detect_score >= detect_score_threshold:
            boxes.append((roi[0], roi[1], detect_score))
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
    :param boxes: list of tuples (x_min <int>, y_min <int>, detection score <float>)
    :param max_boxes: integer, maximum allowed detections
    :param iou_threshold: integer, "intersection over union" threshold used for NMS filtering
    :returns: NMS filtered list of tuples (x_min <int>, y_min <int>, detection score <float>)
    """
    # Break list of tuples down into numpy arrays. (Ugly, but the easiest way) 
    detect_scores = np.array([box[2] for box in boxes])
    boxes_coords = np.array([[box[0],box[1],box[0]+window_size[0],box[1]+window_size[1]] for box in boxes])
    nms_indices = []
    # Use get_iou() to get the list of indices corresponding to boxes you keep
    idxs = np.argsort(detect_scores)
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
    boxes = [(boxes_coords[index,0], boxes_coords[index,1], detect_scores[index]) for index in nms_indices]
    return boxes


def predict(image, boxes, window_size):
    """
    Predictor, calls the classifier on each found letter.
    :param image: input image as PIL image object
    :param boxes: list of tuples (x_min <int>, y_min <int>, detection score <float>)
    :param window_size: size of the sliding window, tuple of integers (x,y)
    :returns: list of tuples (x_min <int>, y_min <int>, class prediction <string>, prediction score <float>)
    """
    classified_boxes = []
    for box in boxes:
        x,y = box[0], box[1]
        prediction, predict_score = classify(image.crop((x,y,x+window_size[0],y+window_size[1])))
        classified_boxes.append((x, y, prediction, predict_score))
    return classified_boxes


def plot(image, classified_boxes, window_size):
    """
    Plots the bounding boxes with class label and prediction score on top of the original image.
    :param image: input image as PIL image object 
    :param classified_boxes: list of tuples (x_min <int>, y_min <int>, class prediction <string>, prediction score <float>)
    """
    fig1 = plt.figure(dpi=200)
    ax1 = fig1.add_subplot(1,1,1) 
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.axis('off')
    for box in classified_boxes:
        x_min, y_min, x_max, y_max = box[0]-.5, box[1]-.5, box[0]+window_size[0]-.5, box[1]+window_size[1]-.5
        prediction, predict_score = box[2], box[3]
        ax1.text(x_min, y_min-3, "%s %d%%" % (prediction, predict_score*100), color="red", fontsize=3)
        x = [x_max, x_max, x_min, x_min, x_max]
        y = [y_max, y_min, y_min, y_max, y_max]
        line, = ax1.plot(x,y,color="red")
        line.set_linewidth(.5)
    return


### TODO!!!
import string
import random 
def classify(crop):
    """
    Dummy classifier. Returns random values! Chooses class "0" if all pixels in the window are white.
    :param crop: PIL image, a slice of the input image
    :returns: predicted class as string, prediction score as a float
    """
    
    # Replace this with an actual classifier!
    
    background_color = 255
    prediction = "0" if np.min(crop)>=background_color else random.choice(string.ascii_lowercase)
    predict_score = random.random()
    return prediction, predict_score
### TODO!!!
    

def main():
    # Load the image
    image = Image.open("./dataset/detection-images/detection-2.jpg")
    # Hyperparameters
    window_size         = (20,20)
    stride              = (1,1)
    detect_score_threshold     = .85 # Minimum detection score to be considered a hit.
    iou_threshold       = .1 # Too high --> false positives occur.
    max_boxes           = 100
    # Ru(i)n everything
    boxes               = scan(image, window_size, stride, detect_score_threshold)
    boxes               = nms(boxes, window_size, max_boxes, iou_threshold)
    classified_boxes    = predict(image, boxes, window_size)
    plot(image, classified_boxes, window_size)


if __name__== "__main__":
    main()