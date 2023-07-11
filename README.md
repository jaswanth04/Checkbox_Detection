# Checkbox detection and corresponding data extraction

This repository provides the code for detecting checkboxes and extracting the data corresponding to the check box which is marked

## Introduction

Any form can have multiple checkboxes, which when ticked, the corresponding information is extracted for using.  

This repository is built to extract the data in the image (taken by a mobile phone or any handheld camera) where checkboxes are marked. In this repo, a first principle computer vision methodology is used, rather than a deep learning methodology, to keep it simple.  
The basics can be picked from Computer Vision and then can be transitioned to Deep Learning methodology.

## Methodology

To extract the check boxes, we need the following:
1. a template image, which is a scanned, non - marked form, from which we are extracting the data
2. a roi json, which contains all the regions of interest (entities which we want to extract), whose bounding boxes are marked and stored in the form of a json
3. a query image - This is the image for which we want to extract the marked entities

We use Image Aligner, to align a query Image to the template. We use SIFT Alignment for the same.  
After Alignment, in order to extract the marked checkboxes, we do the following:
1. Extract the Region of Interests, according to the roi json
2. Perform the following cleaning to see whether the area is marked or not
  a. Perform Adaptive Threshold to Binarize the Image
  b. Apply contours, and remove the contours with area less than some threshold, used to remove small noise
  c. Find horizontal and vertical lines using Morphology and remove the lines
  d. Use Hough Lines, to remove more lines
  e. The mark might be present now, If the non - black count is greater than a threshold, then this ROI is considered as marked, else, it is not considered to be marked

The above methodology can be found [here](https://github.com/jaswanth04/Checkbox_Detection/blob/main/notebooks/data_extraction.ipynb)

## Usage

The only required packages for this to work is numpy, opencv. If opencv is installed, numpy also will be install along. 

```
pip install opencv-python
```

The [test data](https://github.com/jaswanth04/Checkbox_Detection/tree/main/test_data) is given in the repo, which can be used. 

```
python src/extract.py --query test_data/image_snap_1660127120112.jpeg --template test_data/TRF-1.png --roi test_data/TRF-1_annotation.json
```

## Scope for Improvement

The disadvantages are:
1. We need a template image
2. We need the roi previously marked

We can use deep learning to remove the disadvantage. The following are the steps needs to be performed:
1. Use Dewarping methodology for aligning and correcting the image
2. Use Document Layout Analysis to get the marked part
3. Once the marked part is retrieved, we can use OCR to extract the text of the marked area. 

## Feedback

If you have any issues with running the code, or any logic, please raise an issue, or write to jaswanth04@gmail.com

