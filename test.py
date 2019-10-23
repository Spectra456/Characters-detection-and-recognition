import numpy as np
import cv2
import pickle
import time

# Size of kernel for blur
kernel = np.ones((5,5), np.uint8)
# Loading svm model for char prediction
model = pickle.load(open('model/train_russian_ocr.svm', 'rb'))

def rotate_image(image, angle):
  """
  Rotating image
  :param image:
  :param angle:
  :return: rotated image
  """
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def process_image(img, char_min_h, char_max_h, char_min_w, char_max_w):
    """
    Processing image, after this finding boxes and making prediction.
    :param img:
    :param char_min_h:
    :param char_max_h:
    :param char_min_w:
    :param char_max_w:
    :return: boxes, labels
    """
    # Preprocessing image to get binary image for character prediction
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
    labels = []
    for cnt in contours:
        #bound the images
        x,y,weight,height = cv2.boundingRect(cnt)

        # Checking contour for our letter size parameters
        if weight > min_width and weight < max_width and height > min_height and height < max_height:
           # Resize image to prediction input(30x30) and converting it to 1-d array
           letter_image = np.concatenate(cv2.resize(img[y:y+height,x:x+weight],(30,30), interpolation=cv2.INTER_NEAREST))
           #making prediction
           result = model.predict(letter_image.reshape(1, -1))[0]
           boxes.append([x,y, x + weight, y + height])
           labels.append(result)

    return boxes, labels

def draw(image, boxes, labels):
    """
    Drawing boxes and labels on image
    :param image:
    :param boxes:
    :param labels:
    :return: image
    """
    for i in range(len(boxes)):
        cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), 1)
        cv2.putText(image, labels[i], (boxes[i][0], boxes[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image

if __name__ == '__main__':

    start_time = int(round(time.time() * 1000))
    image = cv2.imread(r'C:\Users\Spectra\NIIAS_CHAR_RECOGNITION\assets\222.PNG', 0)
    boxes, labels = process_image(image, 0.35, 0.80, 0.02, 0.15)
    end_time = int(round(time.time() * 1000))
    print('Frame processing time in ms - {}'.format(end_time - start_time))
    image = draw(image, boxes, labels)
    cv2.imshow('OCR', image)
    cv2.waitKey(0)