from params import *
from utils import read_camera_intrinsic, reverse_transforms, print_and_write, norm_layer
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
    ])

    # Adjust K according to resize and center crop transforms   
    adjusted_K = adjust_intrinsic(K, original_image_size, torch.tensor([256, 256]), torch.tensor([224, 224]))

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
    adjusted_K = adjust_intrinsic(torch.tensor(K), original_image_size, torch.tensor([256, 256]), torch.tensor([224, 224]))

    return adjusted_K



def adjust_intrinsic(k, original_size, resized_size, ceter_crop_size):
    # Adjust the intrinsic matrix K according to the transformations resize and center crop
    scale_factor = resized_size / original_size
    k[0, 0] = k[0, 0] * scale_factor[0]  # fx
    k[1, 1] = k[1, 1] * scale_factor[1]  # fy
    k[0, 2] = k[0, 2] * scale_factor[0]  # cx
    k[1, 2] = k[1, 2] * scale_factor[1]  # cy

    crop_offset = (resized_size - ceter_crop_size) / 2
    k[0, 2] = k[0, 2] - crop_offset[0]  # cx
    k[1, 2] = k[1, 2] - crop_offset[1]  # cy

    return k


def compute_relative_transformations(pose1, pose2):
    t1 = pose1[:, 3]
    R1 = pose1[:, :3]
    t2 = pose2[:, 3]
    R2 = pose2[:, :3]

    R1_T = torch.transpose(R1, 0, 1)
    R_relative = torch.matmul(R2, R1_T) # TODO: Check again if moving to KITTI! (maybe should be torch.matmul(R1_T, R2, ))
    
    t_image_2_world_coor = torch.matmul(R1_T, (t2 - t1))
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
        print_and_write(f'rank of ground-truth not 2: {torch.linalg.matrix_rank(F)}')

    return F


def get_F(poses, idx, K):
    R_relative, t_relative = compute_relative_transformations(
        poses[idx], poses[idx+JUMP_FRAMES])
    E = compute_essential(R_relative, t_relative)
    F = compute_fundamental(E, K, K)

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

def last_sing_value_penalty(output):
    # Compute the SVD of the output
    _, S, _ = torch.svd(output)

    # Add a term to the loss that penalizes the smallest singular value being far from zero
    rank_penalty = torch.mean(torch.abs(S[:, -1]))

    return rank_penalty

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

def update_epoch_stats(stats, first_image, second_image, unormalized_label, output, unormalized_output, output_grad, epoch=0):
    if ENFORCE_RANK_2:
        output = make_rank2(output)
        unormalized_output = make_rank2(unormalized_output)

    algebraic_dist_truth, algebraic_dist_pred, algebraic_dist_pred_unormalized, \
    RE1_dist_truth, RE1_dist_pred, RE1_dist_pred_unormalized, \
    SED_dist_truth, SED_dist_pred, SED_dist_pred_unormalized = torch.tensor(0), torch.tensor(0), torch.tensor(0), \
                                                              torch.tensor(0), torch.tensor(0), torch.tensor(0), \
                                                              torch.tensor(0), torch.tensor(0), torch.tensor(0)
    for img_1, img_2, F_truth, F_pred, F_pred_unormalized, F_pred_grad in zip(first_image, second_image, unormalized_label, output, unormalized_output, output_grad):
        epipolar_geo_truth = EpipolarGeometry(img_1,img_2, F_truth)
        epipolar_geo_pred = EpipolarGeometry(img_1,img_2, F_pred)
        epipolar_geo_pred_unormalized = EpipolarGeometry(img_1, img_2, F_pred_unormalized)
        epipolar_geo_pred_grad = EpipolarGeometry(img_1, img_2, F_pred_grad)

        algebraic_dist_truth = algebraic_dist_truth + epipolar_geo_truth.get_mean_algebraic_distance()
        algebraic_dist_pred = algebraic_dist_pred + epipolar_geo_pred_grad.get_mean_algebraic_distance()
        algebraic_dist_pred_unormalized = algebraic_dist_pred_unormalized + epipolar_geo_pred_unormalized.get_mean_algebraic_distance()
        if RE1_DIST:
            RE1_dist_truth = RE1_dist_truth + epipolar_geo_truth.get_RE1_distance()
            RE1_dist_pred = RE1_dist_pred + epipolar_geo_pred_grad.get_RE1_distance()
            RE1_dist_pred_unormalized = RE1_dist_pred_unormalized + epipolar_geo_pred_unormalized.get_RE1_distance()
        if SED_DIST:
            SED_dist_truth = SED_dist_truth + epipolar_geo_truth.get_SED_distance()
            SED_dist_pred = SED_dist_pred + epipolar_geo_pred.get_SED_distance()
            SED_dist_pred_unormalized = SED_dist_pred_unormalized + epipolar_geo_pred_unormalized.get_SED_distance()
    
    stats["algebraic_dist_truth"] = stats["algebraic_dist_truth"] + (algebraic_dist_truth / len(first_image))
    stats["algebraic_dist_pred"] = stats["algebraic_dist_pred"] + (algebraic_dist_pred / len(first_image))
    stats["algebraic_dist_pred_unormalized"] = stats["algebraic_dist_pred_unormalized"] + (algebraic_dist_pred_unormalized / len(first_image))
    if RE1_DIST:
        stats["RE1_dist_truth"] = stats["RE1_dist_truth"] + (RE1_dist_truth / len(first_image))
        stats["RE1_dist_pred"] = stats["RE1_dist_pred"] + (RE1_dist_pred / len(first_image))
        stats["RE1_dist_pred_unormalized"] = stats["RE1_dist_pred_unormalized"] + (RE1_dist_pred_unormalized / len(first_image))
    if SED_DIST:
        stats["SED_dist_truth"] = stats["SED_dist_truth"] + (SED_dist_truth / len(first_image))
        stats["SED_dist_pred"] = stats["SED_dist_pred"] + (SED_dist_pred / len(first_image))
        stats["SED_dist_pred_unormalized"] = stats["SED_dist_pred_unormalized"] + (SED_dist_pred_unormalized / len(first_image))

    if VISIUALIZE["epoch"] == epoch:
        epipolar_geo_pred.visualize(sqResultDir=os.path.join(PLOTS_PATH, VISIUALIZE["dir"]), file_num=stats["file_num"])
        stats["file_num"] = stats["file_num"] + 1
    
    return RE1_dist_pred/len(first_image), SED_dist_pred/len(first_image), algebraic_dist_pred/len(first_image)

class EpipolarGeometry:
    def __init__(self, image1_tensors, image2_tensors, F, sequence_num=None, idx=None):
        self.F = F.view(3, 3)

        # Convert images back to original
        self.image1_numpy = reverse_transforms(image1_tensors)
        self.image2_numpy = reverse_transforms(image2_tensors)

        self.pts1, self.pts2 = self.get_keypoints()

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

        pts1 = torch.tensor([kp1[m.queryIdx].pt for m in self.good], dtype=torch.float32).to(device)
        pts2 = torch.tensor([kp2[m.trainIdx].pt for m in self.good], dtype=torch.float32).to(device)

        pts1 = torch.cat((pts1, torch.ones(pts1.shape[0], 1).to(device)), dim=-1)
        pts2 = torch.cat((pts2, torch.ones(pts2.shape[0], 1).to(device)), dim=-1)

        return pts1, pts2
    
    def algebraic_distance(self, F, pt1, pt2):
        return np.abs(pt2.T.dot(F).dot(pt1))        

    def compute_epipolar_lines(self, F, points):
        """Compute the epipolar_lines for multiple points using vectorized operations."""
        return torch.matmul(F, points.view(-1, 3, 1)).view(-1,3)
    
    def point_2_line_distance(self, point, l):
        l = l.flatten()
        result = abs(np.dot(point, l.T) / np.sqrt(l[0]**2 + l[1]**2))

        return result
    
    def point_2_line_distance_all_points(self, points, lines):
        numerators = abs(torch.sum(lines * points, axis=1))  # Element-wise multiplication and sum over rows
        denominators = torch.sqrt(lines[:, 0]**2 + lines[:, 1]**2)

        return numerators / denominators

    def get_RE1_distance(self):
        inhomogeneous_l1 = self.compute_epipolar_lines(self.F.T, self.pts2)[:,0:2]
        inhomogeneous_l2 = self.compute_epipolar_lines(self.F, self.pts1)[:,0:2]

        denominator = torch.sum(torch.concat((inhomogeneous_l1, inhomogeneous_l2), dim=1) ** 2, dim=1).view(-1,1,1)

        Ri = self.get_algebraic_distance()
        RE1 = (Ri**2) * denominator

        return torch.mean(RE1)
    
    def get_SED_distance(self):
        lines1 = self.compute_epipolar_lines(self.F.T, self.pts2) # shape (n,3)
        lines2 = self.compute_epipolar_lines(self.F, self.pts1)   # shape (n,3)
        
        # Compute the distances from each point to its corresponding epipolar line
        distances1 = self.point_2_line_distance_all_points(self.pts1, lines1)
        distances2 = self.point_2_line_distance_all_points(self.pts2, lines2)

        sed = distances1**2 + distances2**2  # shape (n)

        return torch.mean(sed)
    
    def get_SED_distance2(self):
        lines1 = self.compute_epipolar_lines(self.F.T, self.pts2) # shape (n,3)
        lines2 = self.compute_epipolar_lines(self.F, self.pts1)   # shape (n,3)

        denominator = 1/(lines1[:,0]**2 + lines1[:,1]**2) + 1/(lines2[:,0]**2 + lines2[:,1]**2)

        Ri = self.get_algebraic_distance()
        sed = (Ri**2) * denominator.view(-1,1,1) # shape (n)

        return torch.mean(sed)


    def get_algebraic_distance(self):
        # Returns shape (n,1,1)
        return torch.abs(torch.matmul(torch.matmul(
            self.pts2.view(-1, 1, 3), self.F), self.pts1.view(-1, 3, 1)))

    
    def get_mean_algebraic_distance(self):
        return torch.mean(self.get_algebraic_distance())

    def epipoline(self, x, formula):
        array = formula.flatten()
        a = array[0]
        b = array[1]
        c = array[2]
        return int((-c - a * x) / b)

    def visualize(self, sqResultDir, file_num):
        bad_frames_path = os.path.join(sqResultDir, "bad_frames")
        good_frames_path = os.path.join(sqResultDir, "good_frames")
        os.makedirs(bad_frames_path, exist_ok=True)
        os.makedirs(good_frames_path, exist_ok=True)

        F = self.F.cpu().numpy()
        pts1, pts2 = self.pts1.cpu().numpy(), self.pts2.cpu().numpy()

        img1_line = self.image1_numpy.copy()
        img2_line = self.image2_numpy.copy()

        # drawing epipolar line
        avg_distance_err_img1 = 0
        avg_distance_err_img2 = 0

        img_W = self.image1_numpy.shape[1] - 1
        epip_test_err = 0
        for color_idx, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            x1, y1, _ = pt1
            x2, y2, _ = pt2

            line_1 = np.dot(np.transpose(F), pt2)
            line_2 = np.dot(F, pt1)

            # Get ditance from point to line error
            avg_distance_err_img1 += self.point_2_line_distance(pt1, line_1)
            avg_distance_err_img2 += self.point_2_line_distance(pt2, line_2)
            epip_test_err += self.algebraic_distance(F, pt1, pt2)

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
        SED_dist = self.get_SED_distance().cpu().item()
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
        
        if(SED_dist > 2):
            if MOVE_BAD_IMAGES:
                src_path1 = os.path.join(
                    self.sequence_path, "image_0", self.file_name1)
                dst_path1 = os.path.join(
                    self.sequence_path, "BadFrames", self.file_name1)
                if os.path.exists(src_path1):
                    print(f'moved {src_path1} to {dst_path1}')
                    os.rename(src_path1, dst_path1)
            else:
                cv2.imwrite(os.path.join(sqResultDir, "bad_frames", f'epipoLine_sift_{file_num}.{IMAGE_TYPE}'), vis)
                print(os.path.join(sqResultDir, "bad_frames", f'epipoLine_sift_{file_num}.{IMAGE_TYPE}\n'))

        elif not MOVE_BAD_IMAGES:
            cv2.imwrite(os.path.join(sqResultDir, "good_frames", f'epipoLine_sift_{file_num}.{IMAGE_TYPE}'), vis)
            print(os.path.join(sqResultDir, "good_frames", f'epipoLine_sift_{file_num}.{IMAGE_TYPE}\n'))

        return SED_dist
        



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