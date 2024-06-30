import cv2

img = cv2.imread("sequences/00/image_0/000000.png")  # Use a valid image file path
cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
