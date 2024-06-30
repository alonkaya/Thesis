import cv2
import numpy as np

# Function to draw points on the image
def draw_points(image, point, color=(0, 255, 0)):
    # for point in points:
    print(int(point[0]), int(point[1]))
    # print(image)
    cv2.circle(image, (int(point[0]), int(point[1])), 5, color, -1)

# File path to the image
file_path = "sequences/00/image_0/004134.png"

# Read the image using OpenCV
image_np = cv2.imread(file_path, cv2.IMREAD_COLOR)
# Draw points on the images
points = np.array([47,46])
draw_points(image_np, points)
