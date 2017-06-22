import cv2
import numpy as np

image_size = 120
bbox = {'top': 10, 'left': 50, 'height': 50, 'width': 20}
image_path = './test/420.png'
image = cv2.imread(image_path, 1)
cv2.waitKey(50)
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
orig_height =  list(image.shape)[0]
orig_width =  list(image.shape)[1]

image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
ntop =  (bbox['top'] * image_size) / orig_height
nleft =  (bbox['left'] * image_size) / orig_width
nheight = ((bbox['top'] + bbox['height'])* image_size) / orig_height - ntop
nwidth = ((bbox['left'] + bbox['width'])* image_size) / orig_width - nleft
new_bbox = {'top': ntop, 'left': nleft, 'height': nheight, 'width': nwidth}

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(image, kernel, iterations = 1)
dilate = cv2.dilate(image, kernel, iterations = 1)
cv2.imshow("orig_image", image)
cv2.imshow("erosion", erosion)
cv2.imshow("dilate", dilate)
cv2.waitKey(0)