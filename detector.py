#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt

def sliding_window(image, window_size, stride):
    """
    Takes an input image, then slices off predictions of size $window_size.
    Window moves according to $stride.
    """
    for x in range(0, image.size[0], stride[1]):
        for y in range(0, image.size[1], stride[0]):
            yield (x, y, image.crop((x,y,window_size[0],window_size[1])))

# The predict() function will be replaced by an actual classifier! 
# Delete these lines once the classifier is available!!!
import string
import random 
def predict(window):
    """
    Returns the input image, the prediction and confidence score.
    Just random values right now.
    """
    prediction = random.choice(string.ascii_lowercase)
    confidence = random.random()
    return prediction, confidence


def scan(image, window_size, stride):
    """
    Returns a list of predictions, with their predicted letters and positions.
    """
    predictions = []
    for w, window in enumerate(sliding_window(image, window_size, stride)):
        prediction, confidence = predict(window)
        if confidence >= threshold: #and if prediction != "nothing"
            predictions.append((window[0], window[1], prediction, confidence))
    return predictions


def plot(image,predictions):
    """
    Plots the bounding boxes with class label and confidence score on top of the original image.
    """
    fig1 = plt.figure(dpi=400)
    ax1 = fig1.add_subplot(2,2,1) 
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.axis('off')
    
    for w, window in enumerate(predictions):
        x_min, y_min, x_max, y_max = window[0], window[1],window[0]+window_size[0], window[1]+window_size[1]
        prediction, confidence = window[2], window[3]
        ax1.text(x_min, y_min-3, "%s %d%%" % (prediction, confidence*100), color="red", fontsize=1.5)
        x = [x_max, x_max, x_min, x_min, x_max]
        y = [y_max, y_min, y_min, y_max, y_max]
        line, = ax1.plot(x,y,color="red")
        line.set_linewidth(.5)
    return


image = Image.open("./dataset/detection-images/detection-1.jpg")
window_size = (20,20)
stride = (1,1)
threshold = .9999

predictions = scan(image, window_size, stride)
plot(image, predictions)

#print(len(predictions))

#need number and confidence from classifier
#26 + 1 classes (one class for "no character found!")

"""
obsolete (image rescaling is not required!):
from skimage.transform import pyramid_gaussian
for im_scaled in pyramid_gaussian(image, downscale=scale, multichannel=False):
scale = 1.25
"""