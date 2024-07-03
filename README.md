# Coin-Detector
A Python program that detects coins in images through a pipeline of image processing techniques.
There are two versions of the pipeline. `coin_detection_v2` is an optimised version of the basic `coin_detection_v1`.

# The Pipeline
![image](https://github.com/DannyJung23/Coin-Detector/assets/130985271/a23a1093-42de-45c1-8de2-c6e3fc2131f8)

## Converting RGB Image to Greyscale
- **Version 1 & 2**

Read the RGB input image using the `readRGBImageToSeparatePixelArrays` helper function.
Convert the RGB image to greyscale image using the following equation:
$`g = 0.299*r + 0.587*g + 0.114*b`$

## Image Normalising
- **Version 1 & 2**

Normalise the greyscale image by performing 5~95 percentile based mapping. This contrast stretching increases the dynamic range of intensities, improving visual contrast of the image.

<img src="https://github.com/DannyJung23/Coin-Detector/assets/130985271/434084db-da79-4031-b7ba-dda74b49eacb" width="600" height="300">

## Edge Detection
- **Version 1**

Apply a `3x3 Scharr filter` in horizontal (x) and vertical (y) directions independently to get the edge map. The computed value for each individual pixel should be stored as `float`.
Take the absolute value of the sum between horizontal and vertical direction edge maps to compute the edge strength as the following equation:
$`g_m(x,y) = |g_x(x,y)| + |g_y(x,y)|`$

- **Version 2**

Apply a `3x3 Laplacian filter` in both horizontal (x) and vertical (y) directions to get the edge map. Laplacian filter is used instead of Scharr filter to enhance orientation independence and detection of fine details.
```Python
Laplacian filter = [[1.0, 1.0, 1.0],
                    [1.0, -8.0, 1.0],
                    [1.0, 1.0, 1.0]]
```

## Image Blurring
- **Version 1**

Apply a `5x5 mean filter` to smooth the image by reducing the variation of intensities between neighbouring pixels. However, in this approach, a single outlier can significantly affect the average of all the neighbouring pixels.
The filter is applied 3 times to blur the image properly.

- **Version 2**

Apply a `5x5 median filter` to optimise the reduce of the salt-and-pepper noise. This approach is also robust to outliers since it is a non-linear filter.
The filter is applied 3 times to blur the image properly.

## Image Thresholding
- **Version 1**

A simple thresholding with a single threshold value of 22 is performed to segment the coins into a binary region map. Any pixel value smaller than 22 is set to 0 and any pixel value larger than 22 is set to 255.

- **Version 2**

An adaptive thresholding is used instead of simple thresholding to segment the coins into a binary region map. This accounts for empirical probability distributions of the object and produces optimal threshold value for each input image.
The while loop in the function breaks only when the newly calculated threshold value equals the lastly calculated threshold value, meaning the value is optimal.

## Closing the Image (Dilation & Erosion)
- **Version 1 & 2**

Closing of an image `f` is filling any holes or channels inside an object that are smaller than the structuring element `s`. This is done by performing a set of dilation process followed by perfoming a set of erosion process.

<img src="https://github.com/DannyJung23/Coin-Detector/assets/130985271/3484067a-be64-4c7e-a8b2-81d140ced670" width="500" height="200">

The structuring element used is a circular 5x5 kernel.
```Python
kernel = [[0, 0, 1, 0, 0],
          [0, 1, 1, 1, 0],
          [1, 1, 1, 1, 1],
          [0, 1, 1, 1, 0],
          [0, 0, 1, 0, 0]]
```

## Connected Component Analysis
- **Version 1 & 2**

Connected component analysis finds all connected components in the image to identify and count different coin components. This is done by implementing a `Queue` based function to check the connectedness of each pixel with its 4 neighbouring pixels.
After closing the image from the previous step, there might still be some holes in the binary image which is fine as long as the coin component is one connected component.
If there are isolated islands (non-connected component inside a hole in a coin component), increase the number of dilation and erosion to close the isolated islands in the closing step.

## Outputting the Images
Simply run the program with a desired input image to get an output. Change the directory of an input image by changing `image_name` and `input_filename` to test other images. 
The output image will be automatically saved to the directory assigned to `default_output_path`. Some example input images are included in the `Images` folder.
There are 6 simple cases and 3 complex cases where simple cases only show coins on a pure background and complex cases show coins with other objects and components that need to be distinguished in order to detect coins only.

![image](https://github.com/DannyJung23/Coin-Detector/assets/130985271/d356263e-95a5-44ca-885f-9221e49bf3db)
![image](https://github.com/DannyJung23/Coin-Detector/assets/130985271/ec838d7e-4abd-439e-aa2a-39989a1c43d6)

Examples of the output for the simple image cases.
