import cv2

img = cv2.imread("epipole_lines/bad_frames/epipoLine_sift_39.png")  # Use a valid image file path
cv2.imshow('Test Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
