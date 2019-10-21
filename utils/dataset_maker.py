import numpy as np
import cv2

kernel = np.ones((5,5), np.uint8)

def process_image(img_path, char_min_h, char_max_h, char_min_w, char_max_w):
    """
    Processing image, after this finding boxes.
    :param img:
    :param char_min_h:
    :param char_max_h:
    :param char_min_w:
    :param char_max_w:
    :return: boxes
    """
    # Preprocessing image to get binary image for character prediction
    img = cv2.imread(img_path, 0)
    blur = cv2.GaussianBlur(img,(3, 3),0)
    ret3,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = cv2.erode(img, kernel, iterations=2)
    height,weight = img.shape

    # the next two lines is based on the assumptions that the width of
    # a license plate should be between 2% and 15% of the license plate,
    # and height should be between 35% and 80%
    # this will eliminate some
    character_dimensions = (char_min_h*height, char_max_h*height, char_min_w*weight, char_max_w*weight)
    min_height, max_height, min_width, max_width = character_dimensions

    # Find contours(letters) and make prediction
    _,contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i,cnt in enumerate(contours):
        #bound the images
        x,y,weight,height = cv2.boundingRect(cnt)

        # Checking contour for our letter size parameters
        if weight > min_width and weight < max_width and height > min_height and height < max_height:
           # Resize image to prediction input(30x30) and converting it to 1-d array
           letter_image = cv2.resize(img[y:y + height, x:x + weight], (30, 30), interpolation=cv2.INTER_NEAREST)
           cv2.imwrite('../assets/result/{}.jpg'.format(i), letter_image)



process_image('../assets/test.png', 0.35, 0.80, 0.02, 0.15 )