import os
import cv2
import numpy as np
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from FunMatrix import EpipolarGeometry
from params import EPIPOLAR_THRESHOLD, SED_TRIM_THRESHOLD

def estimate_fundamental_matrix(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < EPIPOLAR_THRESHOLD * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    pts1 = np.hstack([pts1, np.ones((pts1.shape[0], 1))]) # shape (n, 3)
    pts2 = np.hstack([pts2, np.ones((pts2.shape[0], 1))]) # shape (n, 3)
    
    Fgt = np.array([[ 0.0000,  0.0000,  0.0000],
                    [ 0.0000,  0.0000, -0.7071],
                    [ 0.0000,  0.7071,  0.0000]])
    
    # print(pts1.shape)
    pts1, pts2 = trim_by_sed(pts1, pts2, Fgt)
    print(pts1.shape)
    # print()

    return pts1, pts2

def trim_by_sed(pts1, pts2, Fgt, threshold=SED_TRIM_THRESHOLD, min_keypoints=10):       
        # sed = symmetric_epipolar_distance(Fgt, pts1, pts2) # shape (n,)
        sed = get_SED_distance(Fgt, pts1, pts2) # shape (n,)
        
        # print(max(sed))
        sorted_indices = np.argsort(sed)
        
        # Find indices of keypoints with SED values below threshold
        selected_indices = sorted_indices[sed[sorted_indices] < threshold]
        
        # If there are fewer than min_keypoints below threshold, select the smallest min_keypoints keypoints
        if len(selected_indices) < min_keypoints:
            selected_indices = sorted_indices[:min_keypoints]

        # Select corresponding keypoints
        trimmed_pts1 = pts1[selected_indices].reshape(-1, 3)
        trimmed_pts2 = pts2[selected_indices].reshape(-1, 3)
        
        return trimmed_pts1, trimmed_pts2


def get_SED_distance(F, pts1, pts2):
    lines1 = compute_epipolar_lines(F.T, pts2)  # shape (n, 3)
    lines2 = compute_epipolar_lines(F, pts1)    # shape (n, 3)

    # Compute the distances from each point to its corresponding epipolar line
    distances1 = point_2_line_distance_all_points(pts1, lines1) # shape (n)
    distances2 = point_2_line_distance_all_points(pts2, lines2) # shape (n)

    sed = distances1 ** 2 + distances2 ** 2  # shape (n)

    return sed  # shape (n)

def compute_epipolar_lines(F, points):
    # F shape: (3, 3), points shape: (n, 3)
    lines = np.matmul(F, points.T).T  # shape: (n, 3)
    norm = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2 + 1e-8).reshape(-1,1) # shape: (n, 1)

    return lines / norm

def point_2_line_distance_all_points(points, lines):
    # Both points and lines are of shape (n, 3)
    dist = np.abs(np.sum(lines * points, axis=-1))  # Element-wise multiplication and sum over last dimension
    return dist # shape (batch_size, n)


avg_sed = 0
end = 1600
for i in range(end):
    p1 = f'sequences/09/image_0/{i:06}.png'
    p2 = f'sequences/09/image_1/{i:06}.png'
    if not os.path.exists(p1) or not os.path.exists(p2): continue
        
    # Load images
    img1 = cv2.imread(p1)  # Update the path to the image
    img2 = cv2.imread(p2)  # Update the path to the image
    # Estimate the fundamental matrix
    try:
        pts1, pts2 = estimate_fundamental_matrix(img1, img2)
    except Exception as e:
        print(e)
        continue

    # Compute Fundamental Matrix using RANSAC
    F, mask = cv2.findFundamentalMat(pts1[:,:2], pts2[:,:2], cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)   

    pts1 = torch.from_numpy(pts1).float().unsqueeze(0)
    pts2 = torch.from_numpy(pts2).float().unsqueeze(0)
    F = torch.from_numpy(F).float().unsqueeze(0)
    ep = EpipolarGeometry(None, None, F, pts1, pts2)
    sed = ep.get_mean_SED_distance()
    

    # Assuming F_matrix, points1, and points2 are defined as from your previous function call
    sed = np.mean(get_SED_distance(F, pts1, pts2))
    # print("SED:", sed)
    # print()
    avg_sed += sed
print("Average SED:", avg_sed / end)

