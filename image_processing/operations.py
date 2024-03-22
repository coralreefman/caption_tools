from PIL import Image
import numpy as np
from skimage import filters
from skimage.measure import label
import cv2
import matplotlib.pyplot as plt

def resize(image, max_size=2048):

    width, height = image.size

    # Check if resizing is needed
    if width > max_size or height > max_size:
        # Calculate the aspect ratio
        aspect_ratio = float(width) / float(height)

        # Determine the new width and height while maintaining the aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)

        # Resize the image and return the resized version
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image
    else:
        return image
    
def crop_by_percentage(image, amount):

    # percentage works better for images of varying sizes but similar content

    width, height = image.size

    crop_amount_x = width * amount / 100
    crop_amount_y = height * amount / 100
 
    left = round(crop_amount_x / 2)
    top = round(crop_amount_y / 2)
    right = round(width - (crop_amount_x / 2))
    bottom = round(height - (crop_amount_y / 2))

    cropped_img = image.crop((left, top, right, bottom))

    return cropped_img

def solid_background_detection(image, frame_width, percentage_threshold, region_threshold=5):

    if not (isinstance(image, np.ndarray) and len(image.shape) == 2):
        raise ValueError("Input image must be a 2D NumPy array.")
    if frame_width < 1:
        raise ValueError("Frame width must be at least 1.")
    if not (0 <= percentage_threshold <= 100):
        raise ValueError("Percentage threshold must be between 0 and 100.")
    height, width = image.shape
    if 2 * frame_width > width or 2 * frame_width > height:
        raise ValueError("Frame width is too large for the given image dimensions.")

    top = image[:frame_width, :].flatten()
    bottom = image[-frame_width:, :].flatten()
    left = image[:, :frame_width].flatten()
    right = image[:, -frame_width:].flatten()

    frame_pixels = np.concatenate((top, bottom, left, right))

    # Find the most common pixel value in the frame
    most_common_pixel = np.bincount(frame_pixels).argmax()

    # Check for most common pixel condition along the top-bottom and left-right frame
    tb_condition = np.all(top == most_common_pixel) and np.all(bottom == most_common_pixel)
    lr_condition = np.all(left == most_common_pixel) and np.all(right == most_common_pixel)

    # Count the occurrence of the most common pixel
    most_common_pixel_count = np.count_nonzero(frame_pixels == most_common_pixel)

    # Compute the percentage of the most common pixel in the frame
    most_common_pixel_percentage = (most_common_pixel_count / frame_pixels.size) * 100

    # Count the number of contiguous regions of same color
    frame_labels = label(frame_pixels)
    region_count = np.max(frame_labels)

    print(f'region count: {region_count}')

    # Check if conditions are met
    color_condition = most_common_pixel_percentage >= percentage_threshold or tb_condition or lr_condition
    region_condition = region_count <= region_threshold

    if most_common_pixel == 255:
        inverted_img = 255 - image
    else:
        inverted_img = None

    return color_condition and region_condition, inverted_img


def find_largest_rectangle(thresh):
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_area = 0
    largest_rectangle = None
    
    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        
        area = w * h
        
        # check if it doesn't exceed aspect ratio 1:3, hardcoded for now
        aspect_ratio = w / h
        if area > largest_area and 1/3 <= aspect_ratio <= 3:
            largest_area = area
            largest_rectangle = [x, y, w, h]
    
    return largest_rectangle

def crop_from_background(img):

    img_grey = img.convert('L')
    grey = np.array(img_grey)
    # threshold = filters.threshold_isodata(grey)
    # threshold = filters.threshold_triangle(grey)
    threshold = filters.threshold_otsu(grey)
    # threshold = filters.threshold_yen(grey)
    bin_img = grey > threshold
    bw = (bin_img).astype(np.uint8) * 255 
    bw = cv2.medianBlur(bw, 11)

    # plt.imshow(bw)
    # plt.show()

    frame_width = 2
    threshold_percentage = 60
    min_size_percentage = 0.4

    background_detected, inverted_image = solid_background_detection(bw, frame_width, threshold_percentage)
    if inverted_image is not None:
        bw = inverted_image
    
    if background_detected:

        print("FRAME DETECTED")
        width, height = img.size

        try:
            x, y, w, h = find_largest_rectangle(bw)
        except:
            print("couldn't find largest rectangle")
            return img

        min_size = (w >= width * min_size_percentage) and (h >= height * min_size_percentage)
        same_size = (abs(width - w) == 0 and abs(height - h) == 0)

        print(f"w: {w}, h: {h}, width: {width * 0.3}, height: {height * 0.3}, min_size: {min_size}")
        # if im.size and props are the same, skip
        if not same_size and min_size:

            img_array = np.array(img)
            # img_copy = np.copy(img_array)
            # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # plt.imshow(img_copy)
            # plt.show()
            # cropped_img_array = img_array[y1:y2,x1:x2]
            cropped_img_array = img_array[y:y+h, x:x+w]
            # plt.imshow(cropped_img_array)
            # plt.show()
            cropped_img = Image.fromarray(cropped_img_array.astype('uint8'))
            img = cropped_img

            # recursively call function
            # SHOULD DO DIFFERENT SETTINGS ON SECOND PASS SUCH AS USE THE COLOR IMAGE INSTEAD
            return crop_from_background(cropped_img)
        
    return img


# im1 = Image.open('lena.jpg').convert("P").quantize(colors=2, method=Image.Quantize.MEDIANCUT, kmeans=2, palette="ADAPTIVE", dither=Dither.FLOYDSTEINBERG)