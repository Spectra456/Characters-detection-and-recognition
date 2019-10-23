import numpy as np
import cv2
import utils.dataset_maker as d
import glob
# cap = cv2.VideoCapture(r'V:\NIIIAS\TRAINS1.mkv')
#
# #Reading the first frame
# (grabbed, frame) = cap.read()
# counter = 0
# while(cap.isOpened()):
#     counter += 1
#     (grabbed, frame) = cap.read()
#
#     cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#     cv2.imshow('frame', frame)
#
#     cv2.imshow('100.jpg', frame[880:1070,1103:1600] )
#     cv2.waitKey(1)
#
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     d.process_image(frame[880:1070,1103:1600],0.35, 0.80, 0.02, 0.15, counter)
#
# cap.release()
# cv2.destroyAllWindows()

for i,f in enumerate(glob.glob('../assets/*.PNG')):
    print(f)
    img = cv2.imread(f, 0)
    d.process_image(img, 0.35, 0.80, 0.02, 0.15, i)