from params import *
from utils import read_camera_intrinsic, reverse_transforms, print_and_write
import cv2
import os
from scipy.linalg import rq
import numpy as np

def get_intrinsic_REALESTATE(specs_path, original_image_size):
    intrinsics = read_camera_intrinsic(specs_path)
    width = original_image_size[0]
    height = original_image_size[1]

    K = torch.tensor([
        [width*intrinsics[0],     0,          width*intrinsics[2]],
        [0,             height*intrinsics[1],  height*intrinsics[3]],
        [0,                 0,          1]
    ]).to(device)

    # Adjust K according to resize and center crop transforms   
    adjusted_K = adjust_intrinsic(K, original_image_size, torch.tensor([256, 256]).to(device), torch.tensor([224, 224]).to(device))

    return adjusted_K

def get_intrinsic_KITTI(calib_path, original_image_size):
    projection_matrix = read_camera_intrinsic(calib_path).reshape(3,4)

    # Extract the 3x3 part of the matrix (ignoring the last column)
    M = projection_matrix[:, :3]

    # Perform RQ decomposition on M
    K, _ = rq(M)

    # Adjust the signs to ensure the diagonal of K is positive
    T = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, T)

    # Normalize K to ensure the bottom-right value is 1
    K = K / K[2, 2]

    # Adjust K according to resize and center crop transforms and compute ground-truth F matrix
    adjusted_K = adjust_intrinsic(torch.tensor(K).to(device), original_image_size, torch.tensor([256, 256]).to(device), torch.tensor([224, 224]).to(device))

    return adjusted_K



def adjust_intrinsic(k, original_size, resized_size, ceter_crop_size):
    # Adjust the intrinsic matrix K according to the transformations resize and center crop
    scale_factor = resized_size / original_size
    k[0, 0] = k[0, 0] + scale_factor[0]  # fx
    k[1, 1] = k[1, 1] + scale_factor[1]  # fy
    k[0, 2] = k[0, 2] + scale_factor[0]  # cx
    k[1, 2] = k[1, 2] + scale_factor[1]  # cy

    crop_offset = (resized_size - ceter_crop_size) / 2
    k[0, 2] = k[0, 2] + crop_offset[0]  # cx
    k[1, 2] = k[1, 2] + crop_offset[1]  # cy

    return k


def compute_relative_transformations(pose1, pose2):
    t1 = pose1[:, 3]
    R1 = pose1[:, :3]
    t2 = pose2[:, 3]
    R2 = pose2[:, :3]

    R1_T = torch.transpose(R1, 0, 1)
    R_relative = torch.matmul(R2, R1_T)
    
    t_image_2_world_coor = torch.matmul(R1_T, (t2 - t1))
    t_world_2_image_coor = t2 - torch.matmul(R_relative, t1)
    t_relative = t_world_2_image_coor if USE_REALESTATE else t_image_2_world_coor
    
    return R_relative, t_relative


def compute_essential(R, t):
    # Compute the skew-symmetric matrix of t
    t_x = torch.tensor([[0, -t[2], t[1]],
                        [t[2], 0, -t[0]],
                        [-t[1], t[0], 0]]).to(device)

    # Compute the essential matrix E
    E = torch.matmul(t_x, R)
    return E


# Define a function to compute the fundamental matrix F from the essential matrix E and the projection matrices P0 and P1
def compute_fundamental(E, K1, K2):
    K2_inv_T = torch.transpose(torch.linalg.inv(K2), 0, 1)
    K1_inv = torch.linalg.inv(K1)

    # Compute the Fundamental matrix
    F = torch.matmul(K2_inv_T, torch.matmul(E, K1_inv))

    if torch.linalg.matrix_rank(F) != 2:
        print_and_write(f'rank of ground-truth not 2: {torch.linalg.matrix_rank(F)}')

    return F


def get_F(poses, idx, K):
    R_relative, t_relative = compute_relative_transformations(
        poses[idx], poses[idx+JUMP_FRAMES])
    E = compute_essential(R_relative, t_relative)
    F = compute_fundamental(E, K, K)

    return F


def last_sing_value_penalty(output):
    # Compute the SVD of the output
    _, S, _ = torch.svd(output)

    # Add a term to the loss that penalizes the smallest singular value being far from zero
    rank_penalty = torch.mean(torch.abs(S[:, -1]))

    return rank_penalty

def get_avg_epipolar_test_errors(first_image, second_image, unormalized_label, output, unormalized_output):
    # Compute mean epipolar constraint error
    # U1, S1, V1 = torch.svd(output)
    # U2, S2, V2 = torch.svd(unormalized_output)

    # S1[:, -1] = 0
    # S2[:, -1] = 0

    # output = torch.matmul(torch.matmul(U1, torch.diag_embed(S1)), V1.transpose(1, 2))
    # unormalized_output = torch.matmul(torch.matmul(U2, torch.diag_embed(S2)), V2.transpose(1, 2))
    
    avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized = 0, 0, 0
    try:
        for img_1, img_2, F_truth, F_pred, F_pred_unormalized in zip(first_image, second_image, unormalized_label, output, unormalized_output):
            avg_ec_err_truth = avg_ec_err_truth + EpipolarGeometry(img_1,img_2, F_truth).get_epipolar_err()
            avg_ec_err_pred = avg_ec_err_pred + EpipolarGeometry(img_1,img_2, F_pred).get_epipolar_err()
            avg_ec_err_pred_unormalized = avg_ec_err_pred_unormalized + EpipolarGeometry(img_1, img_2, F_pred_unormalized).get_epipolar_err()
    except Exception as e:
        print_and_write(f'Error in get_avg_epipolar_test_errors: {e}')

    avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized = (
        v / len(first_image) for v in (avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized))

    return avg_ec_err_truth, avg_ec_err_pred, avg_ec_err_pred_unormalized

def reconstruction_module(x):
    def get_rotation(rx, ry, rz):
        # normalize input?
        R_x = torch.tensor([
            [1.,    0.,             0.],
            [0.,    torch.cos(rx),    -torch.sin(rx)],
            [0.,    torch.sin(rx),     torch.cos(rx)]
        ], requires_grad=True).to(device)
        R_y = torch.tensor([
            [torch.cos(ry),    0.,    -torch.sin(ry)],
            [0.,            1.,     0.],
            [torch.sin(ry),    0.,     torch.cos(ry)]
        ], requires_grad=True).to(device)
        R_z = torch.tensor([
            [torch.cos(rz),    -torch.sin(rz),    0.],
            [torch.sin(rz),    torch.cos(rz),     0.],
            [0.,            0.,             1.]
        ], requires_grad=True).to(device)
        R = torch.matmul(R_x, torch.matmul(R_y, R_z))
        return R

    def get_inv_intrinsic(f):
        return torch.tensor([
            [-1/(f+1e-8),   0.,             0.],
            [0.,            -1/(f+1e-8),    0.],
            [0.,            0.,             1.]
        ], requires_grad=True).to(device)

    def get_translate(tx, ty, tz):
        return torch.tensor([
            [0.,  -tz, ty],
            [tz,  0,   -tx],
            [-ty, tx,  0]
        ], requires_grad=True).to(device)

    def get_fmat(x):
        # F = K2^(-T)*R*[t]x*K1^(-1)
        # Note: only need out-dim = 8
        K1_inv = get_inv_intrinsic(x[0]) # K1^(-1)
        K2_inv_T = torch.transpose(get_inv_intrinsic(x[1])) # K2^(-T)
        R = get_rotation(x[2], x[3], x[4]) 
        T = get_translate(x[5], x[6], x[7])
        F = torch.matmul(K2_inv_T,
                         torch.matmul(R, torch.matmul(T, K1_inv)))

        return F

    out = get_fmat(x)

    return out


class EpipolarGeometry:
    def __init__(self, image1_tensors, image2_tensors, F, sequence_num=None, idx=None):
        self.F = F.view(3, 3)

        # Convert images back to original
        self.image1_numpy = reverse_transforms(image1_tensors)
        self.image2_numpy = reverse_transforms(image2_tensors)

        self.sequence_path = os.path.join(
            'sequences', sequence_num) if sequence_num else None
        self.file_name1 = f'{idx:06}.{IMAGE_TYPE}' if idx != None else None

        self.colors = [
            (255, 102, 102),
            (102, 255, 255),
            (125, 125, 125),
            (204, 229, 255),
            (0, 0, 204)
        ]

    def get_keypoints(self, threshold=EPIPOLAR_THRESHOLD):
        """Recives numpy images and returns numpy points"""
        # sift = cv2.xfeatures2d.SIFT_create()
        try:
            sift = cv2.SIFT_create()
            bf = cv2.BFMatcher()

            # Detect keypoints and compute descriptors for both images
            (kp1, des1) = sift.detectAndCompute(self.image1_numpy, None)
            (kp2, des2) = sift.detectAndCompute(self.image2_numpy, None)
        except Exception as e:
            print_and_write(f'Error in sift: {e}')
            return

        # matches = bf.match(des1, des2)
        matches = bf.knnMatch(des1, des2, k=2)
        self.good = []
        distances = []
        min_distance_index = 0
        for i, (m, n) in enumerate(matches):
            distances.append(m.distance / n.distance)
            if distances[-1] < threshold / ((len(self.good) // 15)+1):
                self.good.append(m)
            min_distance_index = i if distances[i] < distances[min_distance_index] else min_distance_index
        # If no point passed the threshold, add the smallest distance ratio point
        if len(self.good) == 0:
            self.good.append(matches[min_distance_index][0])

        # Extract the matched keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in self.good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in self.good])

        pts1 = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=-1)
        pts2 = np.concatenate((pts2, np.ones((pts2.shape[0], 1))), axis=-1)

        return pts1, pts2

    def epipolar_test_all_points(self, pts1, pts2):
        # Iterates over all keypoints in 'good'
        errs = torch.abs(torch.matmul(torch.matmul(
            pts2.view(-1, 1, 3), self.F), pts1.view(-1, 3, 1)))

        return torch.mean(errs)

    def get_epipolar_err(self):
        try:
            pts1, pts2 = self.get_keypoints()
        except Exception as e:
            print_and_write(f'Error in get_keypoints: {e}')
            return
        try:
            pts1, pts2 = torch.tensor(pts1, dtype=torch.float32).to(device), torch.tensor(pts2, dtype=torch.float32).to(device)
        except Exception as e:
            print_and_write(f'Error in tensors: {e}')

        try:
            err = self.epipolar_test_all_points(pts1, pts2)
        except Exception as e:
            print_and_write(f'Error in epipolar_test_all_points: {e}')
            return

        return err

    def epipoline(self, x, formula):
        array = formula.flatten()
        a = array[0]
        b = array[1]
        c = array[2]
        return int((-c - a * x) / b)

    def get_point_2_line_error(self, point, l):
        l = l.flatten()
        result = abs(np.dot(point, l.T) / np.sqrt(l[0] * l[0] + l[1] * l[1]))

        return result

    def epipolar_test_single_point(self, pt1, pt2):
        return np.abs(pt2.T.dot(self.F).dot(pt1))

    def visualize(self, sqResultDir, img_idx):
        self.F = self.F.cpu().numpy()

        img1_line = self.image1_numpy.copy()
        img2_line = self.image2_numpy.copy()

        # drawing epipolar line
        avg_distance_err_img1 = 0
        avg_distance_err_img2 = 0

        img_W = self.image1_numpy.shape[1] - 1
        epip_test_err = 0
        pts1, pts2 = self.get_keypoints()
        for color_idx, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            x1, y1, _ = pt1
            x2, y2, _ = pt2

            line_1 = np.dot(np.transpose(self.F), pt2)
            line_2 = np.dot(self.F, pt1)

            # Get ditance from point to line error
            avg_distance_err_img1 += self.get_point_2_line_error(pt2, line_1)
            avg_distance_err_img2 += self.get_point_2_line_error(pt1, line_2)
            epip_test_err += self.epipolar_test_single_point(pt1, pt2)

            # calculating 2 points on the line
            x_0 = self.epipoline(0, line_1)
            x_1 = self.epipoline(img_W, line_1)

            y_0 = self.epipoline(0, line_2)
            y_1 = self.epipoline(img_W, line_2)

            # Set color for line
            color = self.colors[color_idx % len(self.colors)]

            # drawing the line and feature points on the left image
            img1_line = cv2.circle(
                img1_line, (int(x1), int(y1)), radius=4, color=color)
            img1_line = cv2.line(
                img1_line, (0, x_0), (img_W, x_1), color=color, lineType=cv2.LINE_AA)
            # displaying just feature points
            cv2.circle(self.image1_numpy.copy(),
                       (int(x1), int(y1)), radius=4, color=color)

            # drawing the line on the right image
            img2_line = cv2.circle(
                img2_line, (int(x2), int(y2)), radius=4, color=color)
            img2_line = cv2.line(
                img2_line, (0, y_0), (img_W, y_1), color=color, lineType=cv2.LINE_AA)
            # displaying just feature points
            cv2.circle(self.image2_numpy.copy(),
                       (int(x2), int(y2)), radius=4, color=color)

        avg_distance_err_img1 /= pts1.shape[0]
        avg_distance_err_img2 /= pts1.shape[0]
        epip_test_err /= pts1.shape[0]

        vis = np.concatenate((img1_line, img2_line), axis=0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_H = vis.shape[0]
        cv2.putText(vis, str(avg_distance_err_img1), (10, 20), font,
                    0.6, color=(128, 0, 0), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(avg_distance_err_img2), (10, img_H - 10),
                    font, 0.6, color=(0, 0, 128), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(epip_test_err), (10, 210), font, 0.6,
                    color=(130, 0, 150), lineType=cv2.LINE_AA)

        if(avg_distance_err_img1 > 13 or abs(epip_test_err) > 0.01):
            if MOVE_BAD_IMAGES:
                src_path1 = os.path.join(
                    self.sequence_path, "image_0", self.file_name1)
                dst_path1 = os.path.join(
                    self.sequence_path, "BadFrames", self.file_name1)
                if os.path.exists(src_path1):
                    print(f'moved {src_path1} to {dst_path1}')
                    os.rename(src_path1, dst_path1)
            else:
                cv2.imwrite(os.path.join(sqResultDir, "bad_frames", f'epipoLine_sift_{img_idx}.{IMAGE_TYPE}'), vis)
                print(os.path.join(sqResultDir, "bad_frames", f'epipoLine_sift_{img_idx}.{IMAGE_TYPE}\n'))

        elif not MOVE_BAD_IMAGES:
            cv2.imwrite(os.path.join(sqResultDir, "good_frames", f'epipoLine_sift_{img_idx}.{IMAGE_TYPE}'), vis)
            print(os.path.join(sqResultDir, "good_frames", f'epipoLine_sift_{img_idx}.{IMAGE_TYPE}\n'))

