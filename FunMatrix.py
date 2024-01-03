from params import *
import numpy as np
import cv2
from scipy.linalg import rq

# Define a function to read the calib.txt file
def process_calib(calib_path):
    with open(calib_path, 'r') as f:
        p0_matrix = np.array([float(x) for x in f.readline().split()[1:]]).reshape(3, 4)

    return p0_matrix


# Define a function to read the pose files in the poses folder
def read_poses(poses_path):
    poses = []    
    with open(poses_path, 'r') as f:
        for line in f: 
            pose = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
            poses.append(pose)

    return np.stack(poses)


def compute_relative_transformations(pose1, pose2):
    t1 = pose1[:, 3]
    R1 = pose1[:, :3]
    t2 = pose2[:, 3]
    R2 = pose2[:, :3]    

    transposed_R1 = np.transpose(R1)
    R_relative = np.dot(R2, transposed_R1)
    t_relative = np.dot(transposed_R1, (t2 - t1))
    # t_relative = t2 - np.dot(R_relative, t1)

    return R_relative, t_relative

# Define a function to compute the essential matrix E from the relative pose matrix M
def compute_essential(R, t):
    # Compute the skew-symmetric matrix of t
    t_x = np.array([[0, -t[2], t[1]], 
                    [t[2], 0, -t[0]], 
                    [-t[1], t[0], 0]])

    # Compute the essential matrix E
    E = t_x @ R
    return E

# Define a function to compute the fundamental matrix F from the essential matrix E and the projection matrices P0 and P1
def compute_fundamental(E, K1, K2):
    K2_inv_T = np.linalg.inv(K2).T
    K1_inv = np.linalg.inv(K1)
    
    # Compute the Fundamental matrix 
    F = np.dot(K2_inv_T, np.dot(E, K1_inv))

    if not np.linalg.matrix_rank(F) == 2:
        print("rank of ground-truch not 2")

    return F

def get_internal_param_matrix(P):
    # Step 1: Decompose the projection matrix P into the form P = K [R | t]
    M = P[:, :3]
    K, R = rq(M)

    # Enforce positive diagonal for K
    T = np.diag(np.sign(np.diag(K)))
    if np.linalg.det(T) < 0:
        T[1, 1] *= -1

    # Update K and R
    K = np.dot(K, T)
    R = np.dot(T, R)

    K /= K[2, 2]

    return K, R

def check_epipolar_constraint(image1_path, image2_path, F):
    # Load the images
    img1 = cv2.imread(image1_path, 0)
    img2 = cv2.imread(image2_path, 0)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Convert points to homogeneous coordinates
    pts1 = np.concatenate((pts1, np.ones((pts1.shape[0], 1, 1))), axis=-1)
    pts2 = np.concatenate((pts2, np.ones((pts2.shape[0], 1, 1))), axis=-1)

    # Check the epipolar constraint
    for i in range(pts1.shape[0]):
        x = pts1[i, 0, :].reshape(3, 1)
        x_prime = pts2[i, 0, :].reshape(3, 1)
        error = np.dot(np.dot(x_prime.T, F), x)
        print(f'Error for point {i}: {error[0, 0]}')


# # Read the calib.txt file and get the projection matrices
# left_projection_matrix = process_calib('sequences\\00')

# # Compute intrinsic K
# K, _ = get_internal_param_matrix(left_projection_matrix)

# # Read the pose files in the poses folder and get the list of pose matrices
# poses = read_poses('poses\\00.txt')

# # Loop over the pairs of frames
# for i in range(len(poses) - 1):
#     # Compute relative rotation and translation matrices
#     R_relative, t_relative = compute_relative_transformations(poses[i], poses[i+1])

#     # # Compute the essential matrix E
#     E = compute_essential(R_relative, t_relative)

#     # Compute the fundamental matrix F
#     F = compute_fundamental(E, K, K)

#     check_epipolar_constraint(f'sequences\\00\\image_0\\00000{i}.png', f'sequences\\00\\image_0\\00000{i+1}.png', F)
    
