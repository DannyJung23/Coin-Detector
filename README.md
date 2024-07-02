# Coin-Detector
A Python program that detects coins in images through a pipeline of image processing techniques.
There are two versions of the pipeline. `coin_detection_v2` is the optimised version of the basic `coin_detection_v1`.

# The Pipeline
![image](https://github.com/DannyJung23/Coin-Detector/assets/130985271/a23a1093-42de-45c1-8de2-c6e3fc2131f8)

## Converting RGB Image to Greyscale


## Image Normalising

## Edge Detection

## Image Blurring

## Image Thresholding

## Erosion and Dilation

## Connected Component Analysis

## Drawing Bounding Boxes


## Outputting the Images
Simply run the program with a desired input image to get an output. Change the directory of an input image by changing `image_name` and `input_filename` to test other images. 
The output image will be automatically saved to the directory assigned to `default_output_path`. Some example input images are included in the `Images` folder.
There are 6 simple cases and 3 complex cases where simple cases only show coins on a pure background and complex cases show coins with other objects and components that need to be distinguished in order to detect coins only.

![image](https://github.com/DannyJung23/Coin-Detector/assets/130985271/d356263e-95a5-44ca-885f-9221e49bf3db)
![image](https://github.com/DannyJung23/Coin-Detector/assets/130985271/ec838d7e-4abd-439e-aa2a-39989a1c43d6)

Examples of the output for the simple image cases.
