### Coin Detection Version 1 ###

import math
import sys

# Matplotlib will need to be installed if it isn't already
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# Import png reader library
import imageIO.png

TEST_MODE = False # Do not change this variable

def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # Png reader gives width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # Pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# A shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array


### Functions for pipeline v1 ###

class Queue:
    def __init__(self):
        self.items = []
    
    def isEmpty(self):
        return self.items == []
    
    def enqueue(self, item):
        self.items.insert(0, item)
    
    def dequeue(self):
        return self.items.pop()
    
    def size(self):
        return len(self.items)


# Converting from RGB image to greyscale image
def computeRGBToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b):

    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            greyscale_pixel_array[i][j] = round((0.299 * px_array_r[i][j]) + (0.587 * px_array_g[i][j]) + (0.114 * px_array_b[i][j]))

    return greyscale_pixel_array


# Producing cumulative histogram
def computeCumulativeHistogram(pixel_array, image_width, image_height, nr_bins):

    cumulative_histogram = [0] * nr_bins

    for i in range(image_height):
        for j in range(image_width):
            intensity_value = pixel_array[i][j]
            cumulative_histogram[intensity_value] += 1
    
    for i in range(1, nr_bins):
        cumulative_histogram[i] += cumulative_histogram[i-1]
    
    return cumulative_histogram


# Normalising with 5-95 percentile-based mapping
def percentileMapping(pixel_array, image_width, image_height, cumulative_histogram, alpha, beta):
    
    q_alpha = None
    q_beta = None
    total_pixel = image_height * image_width

    alpha_percentile = (alpha/100) * total_pixel
    beta_percentile = (beta/100) * total_pixel

    # Finding q_alpha
    q = 0
    while q_alpha is None:
        if cumulative_histogram[q] > alpha_percentile:
            q_alpha = q
        q += 1
    
    # Finding q_beta
    q = 255
    while q_beta is None:
        if cumulative_histogram[q] < beta_percentile:
            q_beta = q
        q -= 1
    
    # Normalising pixels
    for i in range(image_height):
        for j in range(image_width):
            g = (255 / (q_beta - q_alpha)) * (pixel_array[i][j] - q_alpha)

            if g > 255:
                g = 255
            elif g < 0:
                g = 0
            
            pixel_array[i][j] = round(g)
    
    return pixel_array


# Detecting vertical edges by applying 3x3 Scharr filter in vertical directions
def computeVerticalEdgesScharrAbsolute(pixel_array, image_width, image_height):

    vertical_edge_map = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):

            # BorderIgnore
            if i == 0 or i == image_height-1 or j == 0 or j == image_width-1:
                vertical_edge_map[i][j] = 0.0
            
            # 3x3 Scharr Filter
            else:
                left_column = (3 * pixel_array[i-1][j-1]) + (10 * pixel_array[i][j-1]) + (3 * pixel_array[i+1][j-1])
                right_column = (-3 * pixel_array[i-1][j+1]) + (-10 * pixel_array[i][j+1]) + (-3 * pixel_array[i+1][j+1])
                filtered_value = abs((left_column + right_column) / 32.0)
                vertical_edge_map[i][j] = filtered_value
    
    return vertical_edge_map


# Detecting horizontal edges by applying 3x3 Scharr filter in horizontal directions
def computeHorizontalEdgesScharrAbsolute(pixel_array, image_width, image_height):

    horizontal_edge_map = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):

            # BorderIgnore 3x3
            if i == 0 or i == image_height-1 or j == 0 or j == image_width-1:
                horizontal_edge_map[i][j] = 0.0
            
            # 3x3 Scharr Filter
            else:
                top_row = (3 * pixel_array[i-1][j-1]) + (10 * pixel_array[i-1][j]) + (3 * pixel_array[i-1][j+1])
                bottom_row = (-3 * pixel_array[i+1][j-1]) + (-10 * pixel_array[i+1][j]) + (-3 * pixel_array[i+1][j+1])
                filtered_value = abs((top_row + bottom_row) / 32.0)
                horizontal_edge_map[i][j] = filtered_value
    
    return horizontal_edge_map


# Computing edge strength by adding vertical and horizontal edge maps
def computeEdgeStrength(vertical_edge_map, horizontal_edge_map, image_width, image_height):

    edge_map = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):
            edge_map[i][j] = abs(vertical_edge_map[i][j]) + abs(horizontal_edge_map[i][j])
    
    return edge_map


# Blurring image by applying 5x5 mean filter
def computeMeanFilter5x5(pixel_array, image_width, image_height):

    filtered_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for i in range(image_height):
        for j in range(image_width):

            # BorderIgnore 5x5
            if i < 2 or i > image_height-3 or j < 2 or j > image_width-3:
                filtered_pixel_array[i][j] = 0.0
            
            # 5x5 Mean Filter
            else:
                first_row = (pixel_array[i-2][j-2]) + (pixel_array[i-2][j-1]) + (pixel_array[i-2][j]) + (pixel_array[i-2][j+1]) + (pixel_array[i-2][j+2])
                second_row = (pixel_array[i-1][j-2]) + (pixel_array[i-1][j-1]) + (pixel_array[i-1][j]) + (pixel_array[i-1][j+1]) + (pixel_array[i-1][j+2])
                third_row = (pixel_array[i][j-2]) + (pixel_array[i][j-1]) + (pixel_array[i][j]) + (pixel_array[i][j+1]) + (pixel_array[i][j+2])
                fourth_row = (pixel_array[i+1][j-2]) + (pixel_array[i+1][j-1]) + (pixel_array[i+1][j]) + (pixel_array[i+1][j+1]) + (pixel_array[i+1][j+2])
                fifth_row = (pixel_array[i+2][j-2]) + (pixel_array[i+2][j-1]) + (pixel_array[i+2][j]) + (pixel_array[i+2][j+1]) + (pixel_array[i+2][j+2])
                filtered_value = abs((first_row + second_row + third_row + fourth_row + fifth_row) / 25.0)
                filtered_pixel_array[i][j] = filtered_value
    
    return filtered_pixel_array


# Producing binary region map with simple thresholding
def computeSimpleThreshold(pixel_array, threshold_value, image_width, image_height):

    for i in range(image_height):
        for j in range(image_width):

            # Region 1: Background
            if pixel_array[i][j] < threshold_value:
                pixel_array[i][j] = 0
            
            # Region 2: Coin
            else:
                pixel_array[i][j] = 255
    
    return pixel_array


# Dilating image by applying 5x5 circular structuring element
def computeDilation5x5CircularSE(pixel_array, image_width, image_height):

    dilated_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    # 5x5 Circular SE
    kernel = [[0, 0, 1, 0, 0],
              [0, 1, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 0, 0]]

    # BorderZeroPadding 5x5
    for i in range(image_height):
        pixel_array[i].append(0)
        pixel_array[i].append(0)
        pixel_array[i].insert(0, 0)
        pixel_array[i].insert(0, 0)
    pixel_array.append([0] * (image_width + 4))
    pixel_array.append([0] * (image_width + 4))
    pixel_array.insert(0, ([0] * (image_width + 4)))
    pixel_array.insert(0, ([0] * (image_width + 4)))

    # Hit check with 5x5 Circular SE
    for i in range(2, image_height+2):
        for j in range(2, image_width+2):

            dilated_value = 0
            for k in range(-2, 3):
                for l in range(-2, 3):

                    if pixel_array[i+k][j+l] > 0 and kernel[k+2][l+2] == 1:
                        dilated_value = 255
            
            dilated_pixel_array[i-2][j-2] = dilated_value
    
    return dilated_pixel_array


# Eroding image by applying 5x5 circular structuring element
def computeErosion5x5CircularSE(pixel_array, image_width, image_height):

    eroded_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    # 5x5 Circular SE
    kernel = [[0, 0, 1, 0, 0],
              [0, 1, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 0, 0]]

    # BorderZeroPadding 5x5
    for i in range(image_height):
        pixel_array[i].append(0)
        pixel_array[i].append(0)
        pixel_array[i].insert(0, 0)
        pixel_array[i].insert(0, 0)
    pixel_array.append([0] * (image_width + 4))
    pixel_array.append([0] * (image_width + 4))
    pixel_array.insert(0, ([0] * (image_width + 4)))
    pixel_array.insert(0, ([0] * (image_width + 4)))

    # Fit check with 5x5 Circular SE
    for i in range(2, image_height+2):
        for j in range(2, image_width+2):

            eroded_value = 255
            for k in range(-2, 3):
                for l in range(-2, 3):

                    if pixel_array[i+k][j+l] == 0 and kernel[k+2][l+2] == 1:
                        eroded_value = 0
            
            eroded_pixel_array[i-2][j-2] = eroded_value
    
    return eroded_pixel_array


# Finding all connected components and returning min/max x/y for each component
def computeConnectedComponentLabeling(pixel_array, image_width, image_height):

    labelled_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    label_list = []

    current_label, count = 1, 0
    min_x, min_y, max_x, max_y = image_width-1, image_height-1, 0, 0

    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] > 0 and labelled_pixel_array[i][j] == 0:
                q = Queue()
                q.enqueue([i, j])

                while q.isEmpty() == False:
                    (k, l) = q.dequeue()
                    labelled_pixel_array[k][l] = current_label
                    count += 1
                    min_x = min(min_x, l)
                    min_y = min(min_y, k)
                    max_x = max(max_x, l)
                    max_y = max(max_y, k)

                    # left neighbour
                    if l - 1 >= 0 and pixel_array[k][l-1] > 0 and labelled_pixel_array[k][l-1] == 0:
                        q.enqueue([k, l-1])
                        labelled_pixel_array[k][l-1] = current_label
                    
                    # right neighbour
                    if l + 1 <= image_width-1 and pixel_array[k][l+1] > 0 and labelled_pixel_array[k][l+1] == 0:
                        q.enqueue([k, l+1])
                        labelled_pixel_array[k][l+1] = current_label
                    
                    # lower neighbour
                    if k - 1 >= 0 and pixel_array[k-1][l] > 0 and labelled_pixel_array[k-1][l] == 0:
                        q.enqueue([k-1, l])
                        labelled_pixel_array[k-1][l] = current_label

                    # upper neighbour
                    if k + 1 <= image_height-1 and pixel_array[k+1][l] > 0 and labelled_pixel_array[k+1][l] == 0:
                        q.enqueue([k+1, l])
                        labelled_pixel_array[k+1][l] = current_label
                
                label_list.append([min_x, min_y, max_x, max_y])
                current_label, count = current_label+1, 0
                min_x, min_y, max_x, max_y = image_width-1, image_height-1, 0, 0
    
    return label_list


# Pipeline
def main(input_path, output_path):
    # Change the 'image_name' variable to test other images.
    image_name = 'simple_6'
    input_filename = f'./Images/simple_images/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    # Read in the png file and receive three pixel arrays for red, green and blue components, respectively.
    # Each pixel array contains 8 bit integer values between 0 and 255 encoding the color values.
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)
    
    grey_scale = computeRGBToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)

    nr_bins = 256 # 256 intensity values from 0 to 255

    cumulative_histogram = computeCumulativeHistogram(grey_scale, image_width, image_height, nr_bins)
    
    # 5-95 percentile-based mapping
    print("Converting to greyscale and normalising...")
    normalised_grey_scale = percentileMapping(grey_scale, image_width, image_height, cumulative_histogram, 5, 95)

    print("Detecting vertical edges...")
    vertical_edge_map = computeVerticalEdgesScharrAbsolute(normalised_grey_scale, image_width, image_height)

    print("Detecting horizontal edges...")
    horizontal_edge_map = computeHorizontalEdgesScharrAbsolute(normalised_grey_scale, image_width, image_height)

    print("Producing edge map...")
    edge_map = computeEdgeStrength(vertical_edge_map, horizontal_edge_map, image_width, image_height)

    current_pixel_array = edge_map

    # Blurring image 3 times
    for i in range(3):
        print("Blurring image {} time(s)...".format(i+1))
        mean_filtered = computeMeanFilter5x5(current_pixel_array, image_width, image_height)
        current_pixel_array = mean_filtered

    print("Simple thresholding...")
    binary_region_map = computeSimpleThreshold(current_pixel_array, 22, image_width, image_height)

    current_pixel_array = binary_region_map

    # Dilating image 7 times
    for i in range(7):
        print("Dilating image {} time(s)...".format(i+1))
        dilated_pixel_array = computeDilation5x5CircularSE(current_pixel_array, image_width, image_height)
        current_pixel_array = dilated_pixel_array
    
    # Eroding image 7 times
    for i in range(7):
        print("Eroding image {} time(s)...".format(i+1))
        eroded_pixel_array = computeErosion5x5CircularSE(current_pixel_array, image_width, image_height)
        current_pixel_array = eroded_pixel_array

    print("Performing connected component analysis...")
    bounding_box_list = computeConnectedComponentLabeling(current_pixel_array, image_width, image_height)
    
    
    ### Bounding box coordinates information
    ### bounding_box[0] = min x
    ### bounding_box[1] = min y
    ### bounding_box[2] = max x
    ### bounding_box[3] = max y
    
    # bounding_box_list = [[0, 0, 0, 0]]  # This is a dummy bounding box list. Comment it out when testing.
    # px_array = current_pixel_array # Change this to test current output

    px_array = pyplot.imread(input_filename)
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array, aspect='equal')
    
    # Loop through all bounding boxes
    for bounding_box in bounding_box_list:
        bbox_min_x = bounding_box[0]
        bbox_min_y = bounding_box[1]
        bbox_max_x = bounding_box[2]
        bbox_max_y = bounding_box[3]
        
        bbox_xy = (bbox_min_x, bbox_min_y)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        axs.add_patch(rect)
        
    pyplot.axis('off')
    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_with_bbox.png'
    if not TEST_MODE:
        # Saving output image to the above directory
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)
        
        # Show image with bounding box on the screen
        pyplot.imshow(px_array, cmap='gray', aspect='equal')
        pyplot.show()
    else:
        # Do not change this code block
        pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1
    
    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True
    
    main(input_path, output_path)
    