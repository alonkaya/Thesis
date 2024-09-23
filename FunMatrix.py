from params import *
from utils import read_camera_intrinsic, reverse_transforms, print_and_write, norm_layer, points_histogram, trim
import cv2
import os
from scipy.linalg import rq
import numpy as np

def get_intrinsic_REALESTATE(specs_path, original_image_size):
    intrinsics, _ = read_camera_intrinsic(specs_path)
    width = original_image_size[0]
    height = original_image_size[1]

    K = torch.tensor([
        [width*intrinsics[0],     0,          width*intrinsics[2]],
        [0,             height*intrinsics[1],  height*intrinsics[3]],
        [0,                 0,          1]
    ])

    # Adjust K according to resize and center crop transforms   
    k = adjust_k_resize(K, original_image_size, torch.tensor([RESIZE, RESIZE]))

    center_crop_size = (RESIZE - CROP) // 2
    k = adjust_k_crop(k, center_crop_size, center_crop_size) if not RANDOM_CROP else k

    return k

def get_intrinsic_KITTI(calib_path, original_image_size, adjust_resize=True):
    projection_matrix_cam0, projection_matrix_cam1 = read_camera_intrinsic(calib_path)

    k0, k1 = decompose_k(projection_matrix_cam0.reshape(3, 4)).to(device), decompose_k(projection_matrix_cam1.reshape(3, 4)).to(device)

    if adjust_resize:
        # Adjust K according to resize and center crop transforms and compute ground-truth F matrix
        resized = torch.tensor([RESIZE, RESIZE]).to(device)
        k0, k1 = adjust_k_resize(k0, original_image_size, resized), adjust_k_resize(k1, original_image_size, resized)

    center_crop_size = (RESIZE - CROP) // 2
    k0, k1 = adjust_k_crop(k0, center_crop_size, center_crop_size) if not RANDOM_CROP else k0, adjust_k_crop(k1, center_crop_size, center_crop_size) if not RANDOM_CROP else k1

    return k0, k1

def decompose_k(projection_matrix):
    # Extract the 3x3 part of the matrix (ignoring the last column)
    M = projection_matrix[:, :3].cpu().numpy() 

    # Perform RQ decomposition on M
    K, _ = rq(M)

    # Adjust the signs to ensure the diagonal of K is positive
    T = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, T)

    # Normalize K to ensure the bottom-right value is 1
    K = K / K[2, 2]

    # Ensure K is float32 before converting to PyTorch tensor
    K = K.astype(np.float32)

    return torch.tensor(K, dtype=torch.float32)

def adjust_k_resize(k, original_size, resized_size):
    # Adjust the intrinsic matrix K according to the resize
    scale_factor = resized_size / original_size
    k[0, 0] = k[0, 0] * scale_factor[0]  # fx
    k[1, 1] = k[1, 1] * scale_factor[1]  # fy
    k[0, 2] = k[0, 2] * scale_factor[0]  # cx
    k[1, 2] = k[1, 2] * scale_factor[1]  # cy

    return k

def adjust_k_crop(k, top, left):
    # Adjust the intrinsic matrix K according to the crop
    k[0, 2] = k[0, 2] - left # cx
    k[1, 2] = k[1, 2] - top  # cy

    return k


def compute_relative_transformations(pose1, pose2):
    t1 = pose1[:, 3]
    R1 = pose1[:, :3]
    t2 = pose2[:, 3]
    R2 = pose2[:, :3]

    R1_T = torch.transpose(R1, 0, 1)
    R2_T = torch.transpose(R2, 0, 1)
    R_image_2_world_coor = torch.matmul(R2_T, R1)
    R_world_2_image_coor = torch.matmul(R2, R1_T)
    R_relative = R_world_2_image_coor if USE_REALESTATE else R_image_2_world_coor

    t_image_2_world_coor = torch.matmul(R2_T, (t1 - t2))
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
        U, S, V = torch.svd(F)
        print(f"""rank of gt F not 2: {torch.linalg.matrix_rank(F)}
singular values: {S.cpu().tolist()}\n""")

    return F


def get_F(k0, k1, poses=None, idx=None, jump_frames=JUMP_FRAMES, R_relative=None, t_relative=None):
    if R_relative == None:
        R_relative, t_relative = compute_relative_transformations(poses[idx], poses[idx+jump_frames])
        E = compute_essential(R_relative, t_relative)
    else:
        E = torch.tensor([[ 0,     0,        0],
                          [ 0,     0,       -0.54],
                          [ 0,     0.54,     0]]).to(device)

    F = compute_fundamental(E, k0, k1)

    return F

def pose_to_F(pose, k):
    # compute unormalized_F and F from unormalized_pose and pose

    # TODO: change this if batch size > 1 !! they are originally (-1,3,4)
    pose = pose.view(3, 4)

    R = pose[:, :3]
    t = pose[:, 3]
    E = compute_essential(R, t)
    F = compute_fundamental(E, k, k)

    return F.view(-1,3,3)

def last_sing_value(output):
    # Compute the SVD of the output
    _, S, _ = torch.svd(output)

    # Add a term to the loss that penalizes the smallest singular value being far from zero
    last_sv_sq = torch.mean(torch.abs(S[:, -1])**2)

    return last_sv_sq


def update_distances(img1, img2, F, pts1, pts2):
    epipolar_geo = EpipolarGeometry(img1, img2, F, pts1, pts2)

    algebraic_dist = epipolar_geo.get_mean_algebraic_distance()
    # algebraic_dist_sqr = epipolar_geo.get_sqr_algebraic_distance()
    RE1_dist = epipolar_geo.get_RE1_distance() if RE1_DIST else RE1_dist
    SED_dist = epipolar_geo.get_mean_SED_distance() if SED_DIST else SED_dist
    
    return algebraic_dist, RE1_dist, SED_dist

def update_epoch_stats(stats, img1, img2, label, output, pts1, pts2, plots_path, data_type, epoch=0):
    prefix = "val_" if data_type == "val" else "test_" if data_type == "test" else ""
    
    algebraic_dist_pred, RE1_dist_pred, SED_dist_pred = update_distances(img1, img2, output, pts1, pts2)
 
    stats[f"{prefix}algebraic_pred"] = stats[f"{prefix}algebraic_pred"] + (algebraic_dist_pred.detach())
    # stats[f"{prefix}algebraic_sqr_pred"] = stats[f"{prefix}algebraic_sqr_pred"] + (algebraic_dist_sqr_pred.detach())
    stats[f"{prefix}RE1_pred"] = stats[f"{prefix}RE1_pred"] + (RE1_dist_pred.detach()) if RE1_DIST else stats[f"{prefix}RE1_pred"]
    stats[f"{prefix}SED_pred"] = stats[f"{prefix}SED_pred"] + (SED_dist_pred.detach()) if SED_DIST else stats[f"{prefix}SED_pred"]

    if epoch == 0 or data_type == "test":
        algebraic_dist_truth, RE1_dist_truth, SED_dist_truth = update_distances(img1, img2, label, pts1.detach(), pts2.detach())

        stats[f"{prefix}algebraic_truth"] = stats[f"{prefix}algebraic_truth"] + (algebraic_dist_truth)
        # stats[f"{prefix}algebraic_sqr_truth"] = stats[f"{prefix}algebraic_sqr_truth"] + (algebraic_dist_sqr_truth)
        stats[f"{prefix}RE1_truth"] = stats[f"{prefix}RE1_truth"] + (RE1_dist_truth) if RE1_DIST else stats[f"{prefix}RE1_truth"]
        stats[f"{prefix}SED_truth"] = stats[f"{prefix}SED_truth"] + (SED_dist_truth) if SED_DIST else stats[f"{prefix}SED_truth"]

    return SED_dist_pred


class EpipolarGeometry:
    def __init__(self, image1_tensors, image2_tensors, F, pts1=None, pts2=None):
        self.F = F

        if pts1 is None:
            # Convert images back to original
            self.image1_numpy = reverse_transforms(image1_tensors.cpu(), mean=norm_mean.cpu(), std=norm_std.cpu()) # shape (H, W, 3)
            self.image2_numpy = reverse_transforms(image2_tensors.cpu(), mean=norm_mean.cpu(), std=norm_std.cpu()) # shape (H, W, 3)
            # self.image1_numpy = image1_tensors.cpu().numpy().transpose(1, 2, 0)
            # self.image2_numpy = image2_tensors.cpu().numpy().transpose(1, 2, 0) 
            self.get_keypoints()

        else:
            self.pts1 = pts1
            self.pts2 = pts2

        self.colors = [
            (255, 102, 102),
            (102, 255, 255),
            (125, 125, 125),
            (204, 229, 255),
            (0, 0, 204)
        ]

    def get_keypoints(self, threshold=EPIPOLAR_THRESHOLD):
        """Recives numpy images and returns numpy points"""
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher()

        # Detect keypoints and compute descriptors for both images
        (kp1, des1) = sift.detectAndCompute(self.image1_numpy, None) # input shape (H, W, 3)
        (kp2, des2) = sift.detectAndCompute(self.image2_numpy, None) # input shape (H, W, 3)

        matches = bf.knnMatch(des1, des2, k=2)
        self.good = []
        distances = []
        min_distance_index = 0
        c = 0
        for i, (m, n) in enumerate(matches):
            if n.distance == 0:
                c += 1
                continue
            distances.append(m.distance / n.distance)
            # if distances[-1] < threshold:
            self.good.append(m)
            # min_distance_index = i if distances[i] < distances[min_distance_index] else min_distance_index
        # If no point passed the threshold, add the smallest distance ratio point
        # if len(self.good) == 0:
        #     self.good.append(matches[min_distance_index][0])
        print(c)
        pts1 = torch.tensor([kp1[m.queryIdx].pt for m in self.good], dtype=torch.float32)
        pts2 = torch.tensor([kp2[m.trainIdx].pt for m in self.good], dtype=torch.float32)

        self.pts1 = torch.cat((pts1, torch.ones(pts1.shape[0], 1)), dim=-1).to(device) # shape (n, 3)
        self.pts2 = torch.cat((pts2, torch.ones(pts2.shape[0], 1)), dim=-1).to(device) # shape (n, 3)
        
        # self.pts1, self.pts2 = self.trim_by_sed() ## TODO

    def trim_by_sed(self, threshold=SED_TRIM_THRESHOLD, min_keypoints=5):
        self.pts1, self.pts2 = self.pts1.unsqueeze(0), self.pts2.unsqueeze(0)  # shape (1, n, 3)
        sed = self.get_SED_distance().squeeze(0)                               # shape (n,)
        self.pts1, self.pts2 = self.pts1.squeeze(0), self.pts2.squeeze(0)      # shape (n, 3)

        sorted_indices = torch.argsort(sed)
        
        # Find indices of keypoints with SED values below threshold
        selected_indices = sorted_indices[sed[sorted_indices] < threshold]
        
        # If there are fewer than min_keypoints below threshold, select the smallest min_keypoints keypoints
        if len(selected_indices) < min_keypoints:
            selected_indices = sorted_indices[:min_keypoints]
        
        # Select corresponding keypoints
        trimmed_pts1 = self.pts1[selected_indices].view(-1, 3)
        trimmed_pts2 = self.pts2[selected_indices].view(-1, 3)
        
        return trimmed_pts1, trimmed_pts2
    
    def average_batch(self, errors):
        """
        Input: errors of shape (batch_size * n)
        Output: average error of shape (1)
        """
        # Sum all valid errors
        sum_errors = torch.sum(errors)
        
        # Count non-zero elements
        valid_count = torch.sum(errors != 0).float()

        return sum_errors / (valid_count)

    def get_algebraic_distance(self):
        batch_size, n, _ = self.pts1.shape
        algebraic_distances = torch.abs(torch.matmul(
            torch.matmul(self.pts2.view(batch_size, n, 1, 3), self.F.view(batch_size, 1, 3, 3)),
            self.pts1.view(batch_size, n, 3, 1)))  # Returns shape (batch_size, n, 1, 1)
        return algebraic_distances
    
    def get_mean_algebraic_distance(self):
        algebraic_distances = self.get_algebraic_distance() # shape (batch_size, n, 1, 1)

        return self.average_batch(algebraic_distances.view(-1))
    
    def get_sqr_algebraic_distance(self):
        algebraic_distances = self.get_algebraic_distance() # shape (batch_size, n, 1, 1)
        
        return self.average_batch(algebraic_distances.view(-1) ** 2)

    def get_RE1_distance(self):
        batch_size, num_points, _ = self.pts1.shape

        inhomogeneous_l1 = self.compute_epipolar_lines(self.F.transpose(1, 2), self.pts2)[:, :, 0:2]
        inhomogeneous_l2 = self.compute_epipolar_lines(self.F, self.pts1)[:, :, 0:2]

        denominator = (torch.sum(inhomogeneous_l1 ** 2, dim=2) + torch.sum(inhomogeneous_l2 ** 2, dim=2)).view(batch_size, num_points)

        Ri_sqr = self.get_algebraic_distance().view(batch_size, num_points) ** 2
        RE1 = Ri_sqr / (denominator+1e-8) # shape (batch_size, n)

        return self.average_batch(RE1.view(-1))

    def get_mean_SED_distance(self):
        sed = self.get_SED_distance()   # shape (batch_size, n)

        return self.average_batch(sed.view(-1)) # shape (1)

    def get_SED_distance(self, show_histogram=False, plots_path=None):
        lines1 = self.compute_epipolar_lines(self.F.transpose(1, 2), self.pts2)  # shape (batch_size, n, 3)
        lines2 = self.compute_epipolar_lines(self.F, self.pts1)                  # shape (batch_size, n, 3)

        # Compute the distances from each point to its corresponding epipolar line
        distances1 = self.point_2_line_distance_all_points(self.pts1, lines1) # shape (batch_size, n)
        distances2 = self.point_2_line_distance_all_points(self.pts2, lines2) # shape (batch_size, n)

        sed = distances1 ** 2 + distances2 ** 2  # shape (batch_size, n)
        if show_histogram:
            points_histogram(sed.cpu(), plots_path)

        return sed  # shape (batch_size, n)

    # def get_SED_distance2(self):
    #     lines1 = self.compute_epipolar_lines(self.F.T, self.pts2) # shape (n,3)
    #     lines2 = self.compute_epipolar_lines(self.F, self.pts1)   # shape (n,3)

    #     denominator = 1/(lines1[:,0]**2 + lines1[:,1]**2) + 1/(lines2[:,0]**2 + lines2[:,1]**2)

    #     Ri = self.get_algebraic_distance()
    #     sed = (Ri**2) * denominator.view(-1,1,1) # shape (n)

    #     return sed.view(-1)
    
    def compute_epipolar_lines(self, F, points):
        # F shape: (batch_size, 3, 3), points shape: (batch_size, n, 3)
        lines = torch.bmm(F, points.transpose(1, 2)).transpose(1, 2)  # shape: (batch_size, n, 3)
        norm = torch.sqrt(lines[:, :, 0] ** 2 + lines[:, :, 1] ** 2 + 1e-8).unsqueeze(-1) # shape: (batch_size, n, 1)

        return lines / norm

    def point_2_line_distance_all_points(self, points, lines):
        # Both points and lines are of shape (batch_size, n, 3)
        dist = torch.abs(torch.sum(lines * points, axis=-1))  # Element-wise multiplication and sum over last dimension
        return dist # shape (batch_size, n)
     
    def point_2_line_distance(self, point, l):
        # Both point and line are of shape (3,) #TODO remove norm from here
        return abs(np.sum(l * point) / np.sqrt(l[0]**2 + l[1]**2))
    
    def algebraic_distance_single_point(self, F, pt1, pt2):
        return np.abs(pt2.T.dot(F).dot(pt1))  
    
    def epipoline(self, x, formula):
        array = formula.flatten()
        a = array[0]
        b = array[1]
        c = array[2]
        return int((-c - a * x) / b)

    def visualize(self, idx, epipolar_lines_path=None, sequence_path=None, move_bad_images=False):
        """ Pass epipolar_lines_path when showing epipolar lines otherwise pass seqeunce_path to move bad images"""
        file_name = f'{idx:06}.{IMAGE_TYPE}' if idx != None else None

        F = self.F.cpu().numpy()
        pts1, pts2 = self.pts1.cpu().numpy(), self.pts2.cpu().numpy()
        
        img1_line = self.image1_numpy.copy()
        img2_line = self.image2_numpy.copy()

        # drawing epipolar line
        avg_distance_err_img1 = 0
        avg_distance_err_img2 = 0

        img_W = self.image1_numpy.shape[1] - 1
        epip_test_err = 0
        for color_idx, (pt1, pt2) in enumerate(zip(pts1, pts2)): # pt1, pt2 of shape (3,)   
            x1, y1, _ = pt1
            x2, y2, _ = pt2

            line_1 = np.dot(np.transpose(F), pt2)
            line_2 = np.dot(F, pt1)

            # Get ditance from point to line error
            avg_distance_err_img1 += self.point_2_line_distance(pt1, line_1)
            avg_distance_err_img2 += self.point_2_line_distance(pt2, line_2)
            epip_test_err += self.algebraic_distance_single_point(F, pt1, pt2)

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

        avg_distance_err_img1 /= self.pts1.shape[0]
        avg_distance_err_img2 /= self.pts1.shape[0]
        epip_test_err /= self.pts1.shape[0]

        RE1_dist = self.get_RE1_distance().cpu().item()
        SED_dist = self.get_mean_SED_distance().cpu().item()
        vis = np.concatenate((img1_line, img2_line), axis=0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_H = vis.shape[0]
        cv2.putText(vis, str(avg_distance_err_img1), (5, 20), font,
                    0.6, color=(128, 0, 0), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(avg_distance_err_img2), (5, img_H - 10),
                    font, 0.6, color=(0, 0, 128), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(epip_test_err), (5, 200), font, 0.6,
                    color=(130, 0, 150), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(RE1_dist), (5, 230), font,
                    0.6, color=(130, 0, 150), lineType=cv2.LINE_AA)  
        cv2.putText(vis, str(SED_dist), (5, 260), font,
                    0.6, color=(130, 0, 150), lineType=cv2.LINE_AA)
        
        if SED_dist > SED_BAD_THRESHOLD and move_bad_images:
            move_images(sequence_path, file_name)
        
        dir_name = "good_frames" if SED_dist < SED_BAD_THRESHOLD else "bad_frames"
        epipolar_lines_path = os.path.join("epipole_lines", epipolar_lines_path, dir_name)
        os.makedirs(epipolar_lines_path, exist_ok=True)
        cv2.imwrite(os.path.join(epipolar_lines_path, f'{file_name}'), vis)
        print(os.path.join(epipolar_lines_path, f'{file_name}\n'))

        return SED_dist

def move_images(sequence_path, file_name):
    src_path = os.path.join(sequence_path, "image_0", file_name)
    os.makedirs(os.path.join(sequence_path, "bad_frames"), exist_ok=True)
    dst_path = os.path.join(sequence_path, "bad_frames", file_name)

    if os.path.exists(src_path):
        print(f'moved {src_path} to {dst_path}')
        os.rename(src_path, dst_path)


def paramterization_layer(x, plots_path):
    """
    Constructs a batch of 3x3 fundamental matrices from a batch of 8-element vectors based on the described parametrization.

    Parameters:
    x (torch.Tensor): A tensor of shape (batch_size, 8) where each row is an 8-element vector.
                      The first 6 elements of each vector represent the first two columns
                      of a fundamental matrix, and the last 2 elements are the coefficients for
                      combining these columns to get the third column.

    Returns:
    torch.Tensor: A tensor of shape (batch_size, 3, 3) representing a batch of 3x3 fundamental matrices.
    """

    # Split the tensor into the first two columns (f1, f2) and the coefficients (alpha, beta)
    f1 = x[:, :3]  # First three elements of each vector for the first column
    f2 = x[:, 3:6]  # Next three elements of each vector for the second column
    alpha, beta = x[:, 6].unsqueeze(1), x[:, 7].unsqueeze(1)  # Last two elements of each vector for the coefficients

    # Compute the third column as a linear combination: f3 = alpha * f1 + beta * f2
    # We need to use broadcasting to correctly multiply the coefficients with the columns
    f3 = alpha * f1 + beta * f2

    # Construct the batch of 3x3 fundamental matrices
    # We need to reshape the columns to concatenate them correctly
    F = torch.cat((f1.view(-1, 3, 1), f2.view(-1, 3, 1), f3.view(-1, 3, 1)), dim=-1)
    for f in F:
        if torch.linalg.matrix_rank(f) != 2:
            U, S, V = torch.svd(f)
            print_and_write(f"""rank of estimated F not 2: {torch.linalg.matrix_rank(f)}                            
singular values: {S.cpu().tolist()}
{f}\n\n""", plots_path)

    return F