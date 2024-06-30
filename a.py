import cv2

img = cv2.imread('test_image.jpg')  # Use a valid image file path
cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
