from retinaface import RetinaFace
import cv2 
import numpy as np
from PIL import Image

'''

TO DO: 
- create function to only say if any faces are in picture or not
- create function to check if there is a face of a certain size present
- potentially draw stuff on the eyes
- potentially crop out only faces or only eyes: 
    - this one needs to ideally sort them when saving, for example based on location,
    so it names e.g. the one with left top coordinate closest to 0 as 001 and the one furthest away as 005

'''

def landmarks_to_int(landmarks):

    landmarks_integers = tuple(int(item) for item in landmarks)

    return landmarks_integers

def convert_pil_to_cv(image):

    image.convert('RGB') 
    img_cv = np.array(image) 

    # If the image has four channels (like an RGBA PNG), take only the first three
    if img_cv.shape[2] == 4:
        img_cv = img_cv[:, :, :3]
    # Convert RGB to BGR 
    img_cv = img_cv[:, :, ::-1].copy() 

    return img_cv

def convert_cv_to_pil(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    return img_pil

def detect_faces(image, threshold):

    img = convert_pil_to_cv(image)
    faces = RetinaFace.detect_faces(img, threshold=threshold)

    return faces

def draw_squares(image, faces):

    img = convert_pil_to_cv(image)

    for f in faces:

        if "face" in f:

            face = faces[f]
            facial_area = face["facial_area"]
            #highlight facial area
            cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)

    return convert_cv_to_pil(img)

def blur_faces(image, faces):

    img = convert_pil_to_cv(image)

    # Create a mask
    mask = np.zeros_like(img)

    for f in faces:

        if "face" in f:

            face = faces[f]
            facial_area = face["facial_area"]

            x1 = facial_area[2]  # left point
            y1 = facial_area[3]  # top point
            x2 = facial_area[0]  # right point
            y2 = facial_area[1]

            # Draw a white, filled rectangle on the mask
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
    # Calculate the kernel size for the blur based on the image dimensions
    kernel_size = max(img.shape[0], img.shape[1]) // 20

    # Make sure kernel size is odd
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    kernel_size = (kernel_size, kernel_size)

    # Blur the mask
    mask = cv2.GaussianBlur(mask, kernel_size, 0)

    # Create a blurred version of the original image
    blurred_img = cv2.GaussianBlur(img, kernel_size, 0)

    # Convert mask and images to float32 type
    mask = mask.astype('float32')/255
    img = img.astype('float32')/255
    blurred_img = blurred_img.astype('float32')/255

    # Blend the original image with the blurred image
    blended = (1 - mask) * img + mask * blurred_img

    # Convert blended image back to uint8 type
    blended = (blended * 255).astype('uint8')

    # Convert BGR to RGB
    # blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    blended = convert_cv_to_pil(blended)

    return blended

def draw_landmarks(image, faces):

    img = convert_pil_to_cv(image)

    for f in faces:

        if "face" in faces:

            face = faces[f]
            facial_area = face["landmarks"]


'''
    detector = MTCNN()
    detections = detector.detect_faces(img)

    print(f"[DEBUG]: detecting faces, found: {len(detections)}")
 
    for detection in detections:

        score = detection["confidence"]
        if score > 0.90:
            x, y, w, h = detection["box"]
            top_left = (x, y)
            bottom_right = (x+w, y+h)

            # detected_face = img[int(y):int(y+h), int(x):int(x+w)]
            cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 1)
'''