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
    k = adjust_k_resize(K, original_image_size, torch.tensor([256, 256]))
    k = adjust_k_crop(k, 16, 16) if not RANDOM_CROP else k

    return k

def get_intrinsic_KITTI(calib_path, original_image_size):
    projection_matrix_cam0, projection_matrix_cam1 = read_camera_intrinsic(calib_path)

    k0, k1 = decompose_k(projection_matrix_cam0.reshape(3, 4)), decompose_k(projection_matrix_cam1.reshape(3, 4))

    # Adjust K according to resize and center crop transforms and compute ground-truth F matrix
    k0, k1 = adjust_k_resize(k0, original_image_size, torch.tensor([256, 256])), adjust_k_resize(k1, original_image_size, torch.tensor([256, 256]))
    k0, k1 = adjust_k_crop(k0, 16, 16) if not RANDOM_CROP else k0, adjust_k_crop(k1, 16, 16) if not RANDOM_CROP else k1

    return k0, k1

def decompose_k(projection_matrix):
    # Extract the 3x3 part of the matrix (ignoring the last column)
    M = projection_matrix[:, :3]

    # Perform RQ decomposition on M
    K, _ = rq(M)

    # Adjust the signs to ensure the diagonal of K is positive
    T = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, T)

    # Normalize K to ensure the bottom-right value is 1
    K = K / K[2, 2]

    return torch.from_numpy(K)

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


def compute_essential(R, t, to_device=False):
    # Compute the skew-symmetric matrix of t
    t_x = torch.tensor([[0, -t[2], t[1]],
                        [t[2], 0, -t[0]],
                        [-t[1], t[0], 0]])
    if to_device: t_x = t_x.to(device)

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
        print(f"""rank of estimated F not 2: {torch.linalg.matrix_rank(F)}
singular values: {S.cpu().tolist()}\n""")

    return F


def get_F(poses, idx, k1, k2, jump_frames=JUMP_FRAMES):
    R_relative, t_relative = compute_relative_transformations(
        poses[idx], poses[idx+jump_frames])
        # torch.tensor(poses[idx][:3,:], dtype=torch.float32), torch.tensor(poses[idx+jump_frames][:3,:], dtype=torch.float32))
    # R_relative = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32)
    # t_relative = torch.tensor([0.54, 0, 0], dtype=torch.float32)
    E = compute_essential(R_relative, t_relative)
    F = compute_fundamental(E, k1, k2)

    return F

def pose_to_F(pose, k):
    # compute unormalized_F and F from unormalized_pose and pose

    # TODO: change this if batch size > 1 !! they are originally (-1,3,4)
    pose = pose.view(3, 4)

    R = pose[:, :3]
    t = pose[:, 3]
    E = compute_essential(R, t, to_device=True)
    F = compute_fundamental(E, k, k)

    return F.view(-1,3,3)

def last_sing_value(output):
    # Compute the SVD of the output
    _, S, _ = torch.svd(output)

    # Add a term to the loss that penalizes the smallest singular value being far from zero
    last_sv_sq = torch.mean(torch.abs(S[:, -1])**2)

    return last_sv_sq

def make_rank2(F, is_batch=True):
    U1, S1, Vt1 = torch.linalg.svd(F, full_matrices=False)

    if is_batch:
        S1[:, -1] = 0
    else:
        S1[-1] = 0

    output = torch.matmul(torch.matmul(U1, torch.diag_embed(S1)), Vt1)

    if torch.linalg.matrix_rank(output) != 2:
        print(f'rank of ground-truth not 2: {torch.linalg.matrix_rank(F)}')
    return output

def update_distances(img_1, img_2, F, algebraic_dist, RE1_dist, SED_dist, pts1, pts2):
    epipolar_geo = EpipolarGeometry(img_1, img_2, F, pts1, pts2)
    algebraic_dist = algebraic_dist + epipolar_geo.get_sqr_algebraic_distance()
    RE1_dist = RE1_dist + epipolar_geo.get_RE1_distance() if RE1_DIST else RE1_dist
    SED_dist = SED_dist + epipolar_geo.get_mean_SED_distance() if SED_DIST else SED_dist
    return algebraic_dist, RE1_dist, SED_dist

def update_epoch_stats(stats, first_image, second_image, label, output, pts1_batch, pts2_batch, plots_path, epoch=0, val=False):
    # TODO: change from squared to abs in evalutation
    algebraic_dist_truth, algebraic_dist_pred, \
    RE1_dist_truth, RE1_dist_pred, \
    SED_dist_truth, SED_dist_pred = torch.tensor(0), torch.tensor(0), torch.tensor(0), \
                                    torch.tensor(0), torch.tensor(0), torch.tensor(0)
    for img_1, img_2, F_truth, F_pred, pts1, pts2 in zip(first_image, second_image, label, output, pts1_batch, pts2_batch):
        algebraic_dist_pred, RE1_dist_pred, SED_dist_pred = update_distances(img_1, img_2, F_pred, algebraic_dist_pred, RE1_dist_pred, SED_dist_pred, pts1, pts2)

        if epoch == 0:
            algebraic_dist_truth, RE1_dist_truth, SED_dist_truth = update_distances(img_1, img_2, F_truth, algebraic_dist_truth, RE1_dist_truth, SED_dist_truth, pts1, pts2)

        if epoch == VISIUALIZE["epoch"] and val:
            epipolar_geo_pred = EpipolarGeometry(img_1, img_2, F_pred.detach(), pts1, pts2)
            epipolar_geo_pred.visualize(idx=stats["file_num"], lines_path=os.path.join(plots_path, VISIUALIZE["dir"]))
            stats["file_num"] = stats["file_num"] + 1
 
    algebraic_dist_pred, RE1_dist_pred, SED_dist_pred, algebraic_dist_truth, RE1_dist_truth, SED_dist_truth =\
        (v/len(first_image) for v in [algebraic_dist_pred, RE1_dist_pred, SED_dist_pred, algebraic_dist_truth, RE1_dist_truth, SED_dist_truth])
    
    prefix = "val_" if val else ""
    stats[f"{prefix}algebraic_pred"] = stats[f"{prefix}algebraic_pred"] + (algebraic_dist_pred)
    stats[f"{prefix}RE1_pred"] = stats[f"{prefix}RE1_pred"] + (RE1_dist_pred) if RE1_DIST else stats[f"{prefix}RE1_pred"]
    stats[f"{prefix}SED_pred"] = stats[f"{prefix}SED_pred"] + (SED_dist_pred) if SED_DIST else stats[f"{prefix}SED_pred"]
    stats[f"{prefix}algebraic_truth"] = stats[f"{prefix}algebraic_truth"] + (algebraic_dist_truth)
    stats[f"{prefix}RE1_truth"] = stats[f"{prefix}RE1_truth"] + (RE1_dist_truth) if RE1_DIST else stats[f"{prefix}RE1_truth"]
    stats[f"{prefix}SED_truth"] = stats[f"{prefix}SED_truth"] + (SED_dist_truth) if SED_DIST else stats[f"{prefix}SED_truth"]

    return algebraic_dist_pred, RE1_dist_pred, SED_dist_pred

class EpipolarGeometry:
    def __init__(self, image1_tensors, image2_tensors, F, pts1=None, pts2=None):
        self.F = F.view(3, 3)

        # Convert images back to original
        self.image1_numpy = reverse_transforms(image1_tensors.cpu(), mean=norm_mean, std=norm_std)
        self.image2_numpy = reverse_transforms(image2_tensors.cpu(), mean=norm_mean, std=norm_std)

        if pts1 is None:
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
        # sift = cv2.xfeatures2d.SIFT_create()
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher()

        # Detect keypoints and compute descriptors for both images
        (kp1, des1) = sift.detectAndCompute(self.image1_numpy, None)
        (kp2, des2) = sift.detectAndCompute(self.image2_numpy, None)

        matches = bf.knnMatch(des1, des2, k=2)
        self.good = []
        distances = []
        min_distance_index = 0
        for i, (m, n) in enumerate(matches):
            distances.append(m.distance / n.distance)
            if distances[-1] < threshold:
            # if distances[-1] < threshold / ((len(self.good) // 15)+1):
                self.good.append(m)
            min_distance_index = i if distances[i] < distances[min_distance_index] else min_distance_index
        # If no point passed the threshold, add the smallest distance ratio point
        if len(self.good) == 0:
            self.good.append(matches[min_distance_index][0])

        pts1 = torch.tensor([kp1[m.queryIdx].pt for m in self.good], dtype=torch.float32)
        pts2 = torch.tensor([kp2[m.trainIdx].pt for m in self.good], dtype=torch.float32)

        self.pts1 = torch.cat((pts1, torch.ones(pts1.shape[0], 1)), dim=-1)
        self.pts2 = torch.cat((pts2, torch.ones(pts2.shape[0], 1)), dim=-1)

        self.pts1, self.pts2 = self.trim_by_sed()
      

    def compute_epipolar_lines(self, F, points):
        """Compute the epipolar_lines for multiple points using vectorized operations."""
        return torch.matmul(F, points.view(-1, 3, 1)).view(-1,3)
    
    def point_2_line_distance(self, point, l):
        # Both point and line are 3x1 vectors
        l = l.flatten() # shape (3,)
        return abs(np.dot(l, point) / np.sqrt(l[0]**2 + l[1]**2))
        
    def point_2_line_distance_all_points(self, points, lines):
        numerators = abs(torch.sum(lines * points, axis=1))  # Element-wise multiplication and sum over rows
        denominators = torch.sqrt(lines[:, 0]**2 + lines[:, 1]**2)

        return numerators / denominators

    def get_mean_SED_distance(self):
        return torch.mean(self.get_SED_distance())


    def algebraic_distance_single_point(self, F, pt1, pt2):
        return np.abs(pt2.T.dot(F).dot(pt1))  
    
    def get_algebraic_distance(self):
        return torch.abs(torch.matmul(torch.matmul(
            self.pts2.view(-1, 1, 3), self.F), self.pts1.view(-1, 3, 1))) # Returns shape (n,1,1)

    def get_mean_algebraic_distance(self):
        return torch.mean(self.get_algebraic_distance())
    
    def get_sqr_algebraic_distance(self):
        return torch.mean(self.get_algebraic_distance()**2)
    
    
    def get_RE1_distance(self):
        inhomogeneous_l1 = self.compute_epipolar_lines(self.F.T, self.pts2)[:,0:2]
        inhomogeneous_l2 = self.compute_epipolar_lines(self.F, self.pts1)[:,0:2]

        denominator = (torch.sum(inhomogeneous_l1**2, dim=1) + torch.sum(inhomogeneous_l2**2, dim=1)).view(-1,1,1)

        Ri = self.get_algebraic_distance()
        RE1 = (Ri**2) / denominator

        return torch.mean(RE1)


    def trim_by_sed(self, threshold=SED_TRIM_THRESHOLD):
        sed = self.get_SED_distance()
        return self.pts1[sed < threshold].view(-1,3), self.pts2[sed < threshold].view(-1,3)

    def get_SED_distance(self, show_histogram=False, plots_path=None):
        lines1 = self.compute_epipolar_lines(self.F.T, self.pts2) # shape (n,3)
        lines2 = self.compute_epipolar_lines(self.F, self.pts1)   # shape (n,3)

        # Compute the distances from each point to its corresponding epipolar line
        distances1 = self.point_2_line_distance_all_points(self.pts1, lines1)
        distances2 = self.point_2_line_distance_all_points(self.pts2, lines2)

        sed = distances1**2 + distances2**2  # shape (n)

        if show_histogram:
            points_histogram(sed.cpu(), plots_path)

        return sed
    
    def get_SED_distance2(self):
        lines1 = self.compute_epipolar_lines(self.F.T, self.pts2) # shape (n,3)
        lines2 = self.compute_epipolar_lines(self.F, self.pts1)   # shape (n,3)

        denominator = 1/(lines1[:,0]**2 + lines1[:,1]**2) + 1/(lines2[:,0]**2 + lines2[:,1]**2)

        Ri = self.get_algebraic_distance()
        sed = (Ri**2) * denominator.view(-1,1,1) # shape (n)

        return torch.mean(sed)


    def epipoline(self, x, formula):
        array = formula.flatten()
        a = array[0]
        b = array[1]
        c = array[2]
        return int((-c - a * x) / b)

    def visualize(self, idx, epipolar_lines_path=None, sequence_path=None, move_bad_images=False):
        """ Pass lines_path when showing epipolar lines otherwise pass seqeunce_path to move bad images"""
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

            line_1 = np.dot(F, pt2)
            line_2 = np.dot(np.transpose(F), pt1)

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
        epipolar_lines_path = os.path.join(epipolar_lines_path, dir_name)
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