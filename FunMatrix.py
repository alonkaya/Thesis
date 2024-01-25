from params import *
from utils import read_calib
import cv2
import os
from scipy.linalg import rq
import numpy as np

def get_intrinsic(calib_path):
    projection_matrix = read_calib(calib_path)

    # TODO: check if this func is correct
    # Step 1: Decompose the projection matrix P into the form P = K [R | t]
    M = projection_matrix[:, :3]
    K, R = rq(M)
    K = torch.tensor(K).to(device)

    # Enforce positive diagonal for K
    T = torch.diag(torch.sign(torch.diag(K)))
    if torch.det(T) < 0:
        T[1, 1] *= -1

    # Update K and R
    K = torch.matmul(K.clone(), T)
    # R = torch.matmul(T, R)

    last_elem = K[2, 2]
    K /= last_elem.clone()

    return K

def adjust_intrinsic(k, original_size, resized_size, ceter_crop_size):
    # Adjust the intrinsic matrix K according to the transformations resize and center crop
    scale_factor = resized_size / original_size
    k[0, 0] *= scale_factor[0]  # fx
    k[1, 1] *= scale_factor[1]  # fy
    k[0, 2] *= scale_factor[0]  # cx
    k[1, 2] *= scale_factor[1]  # cy

    crop_offset = (resized_size - ceter_crop_size) / 2
    k[0, 2] -= crop_offset[0]  # cx
    k[1, 2] -= crop_offset[1]  # cy

    return k

def compute_relative_transformations(pose1, pose2):
    t1 = pose1[:, 3]
    R1 = pose1[:, :3]
    t2 = pose2[:, 3]
    R2 = pose2[:, :3]    

    transposed_R1 = torch.transpose(R1, 0, 1)
    R_relative = torch.matmul(R2, transposed_R1)
    t_relative = torch.matmul(transposed_R1, (t2 - t1))
    # t_relative = t2 - np.dot(R_relative, t1)

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

    if  torch.linalg.matrix_rank(F) != 2:
        print("rank of ground-truch not 2")

    return F

def get_F(poses, idx, K):
    R_relative, t_relative = compute_relative_transformations(poses[idx], poses[idx+jump_frames])
    E = compute_essential(R_relative, t_relative)
    F = compute_fundamental(E, K, K)
    
    return F

def last_sing_value_penalty(output):
    # Compute the SVD of the output
    _, S, _ = torch.svd(output)
    
    # Add a term to the loss that penalizes the smallest singular value being far from zero
    rank_penalty = torch.mean(torch.abs(S[:,-1]))

    return rank_penalty

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
            # TODO: What about the proncipal points? 
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
            K1_inv = get_inv_intrinsic(x[0])
            K2_inv = get_inv_intrinsic(x[1]) #TODO: K2 should be -t not just -1..
            R  = get_rotation(x[2], x[3], x[4])
            T  = get_translate(x[5], x[6], x[7])
            F  = torch.matmul(K2_inv,
                    torch.matmul(R, torch.matmul(T, K1_inv)))

            # to get the last row as linear combination of first two rows
            # new_F = get_linear_comb(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])
            # new_F = get_linear_comb(flat[0], flat[1], flat[2], flat[3], flat[4], flat[5], x[6], x[7])
            # flat = tf.reshape(new_F, [-1])
            return F

        out = get_fmat(x)

        return out

class EpipolarGeometry:
    def __init__(self, image1_tensor, image2_tensor, F, sequence_num=None, idx=None):
        self.F = F.view(3, 3)
        # Recsale pixels to original size [0,1] -> [0,255]
        # self.img1 = (image1_tensor.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        # self.img2 = (image2_tensor.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8) 
        
        self.img1 = (image1_tensor.permute(1, 2, 0) * 255).to(torch.uint8)
        self.img2 = (image2_tensor.permute(1, 2, 0) * 255).to(torch.uint8)

        self.sequence_path = os.path.join('sequences', sequence_num) if sequence_num else None
        self.file_name1 = f'{idx:06}.png'if idx else None

        self.colors = [
            (255, 102, 102),
            (102, 255, 255),
            (125, 125, 125),
            (204, 229, 255),
            (0, 0, 204)
        ]


    def get_keypoints(self, threshold=epipolar_constraint_threshold):
        # sift = cv2.xfeatures2d.SIFT_create()
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher()

        # TODO: make sure img1,img2 are grayscale
        # self.img1 = cv2.cvtColor(self.img1.copy(), cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors for both images
        (kp1, des1) = sift.detectAndCompute(np.array(self.img1.cpu()), None)
        (kp2, des2) = sift.detectAndCompute(np.array(self.img2.cpu()), None)

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
        pts1 = torch.FloatTensor([kp1[m.queryIdx].pt for m in self.good])
        pts2 = torch.FloatTensor([kp2[m.trainIdx].pt for m in self.good])

        pts1 = torch.cat((pts1, torch.ones((pts1.shape[0], 1))), dim=-1).to(device)
        pts2 = torch.cat((pts2, torch.ones((pts2.shape[0], 1))), dim=-1).to(device)

        return pts1, pts2

    def epipolar_test_single_point(self, pt1, pt2): 
        return abs(torch.matmul(torch.matmul(torch.transpose(pt2, 0, 1), self.F), pt1))
    
    def epipolar_test_avg_points(self, pts1, pts2):
        # Iterates over all keypoints in 'good'
        print(pts2.shape)
        # errs = abs(torch.matmul(torch.matmul(pts2, self.F), pts1.reshape(-1,)))
        temp = torch.matmul(pts2.view(-1, 1, pts2.shape[-1]), self.F)

        # Perform batch matrix multiplication with pts1 and take absolute value
        errs = torch.abs(torch.matmul(temp, pts1.view(-1, pts1.shape[-1], 1)))        
        avg_err = torch.mean(errs)
        # return avg_err

        error = 0
        for (pt1, pt2) in zip(pts1, pts2):
            error += self.epipolar_test_single_point(pt1, pt2)
        avg_err2 = error / pts1.shape[0]
        print(avg_err, avg_err2)
        return avg_err

    def get_epipolar_err(self):
        pts1, pts2 = self.get_keypoints()
        return self.epipolar_test_avg_points(pts1, pts2)
    
    def epipoline(self, x, formula):
        array = formula.flatten()
        a = array[0]
        b = array[1]
        c = array[2]
        return int((-c - a * x) / b)
        
    def get_point_2_line_error(self, point, l):
        l = l.flatten()
        result = abs(torch.matmul(point, torch.transpose(l)) / torch.sqrt(l[0] * l[0] + l[1] * l[1]))
        
        return result 
    
    def visualize(self, sqResultDir, img_idx):
        img1_line = self.img1.copy()
        img2_line = self.img2.copy()

        # drawing epipolar line
        avg_distance_err_img1 = 0
        avg_distance_err_img2 = 0

        img_W = self.img1.shape[1] - 1
        epip_test_err = 0
        pts1, pts2 = self.get_keypoints()
        for color_idx, (pt1, pt2) in enumerate(zip(pts1, pts2)):
            x1, y1, _ = pt1
            x2, y2, _ = pt2

            line_1 = torch.matmul(torch.transpose(self.F), pt2)
            line_2 = torch.matmul(self.F, pt1)

            # Get ditance from point to line error
            avg_distance_err_img1 += self.get_point_2_line_error(pt2, line_1)
            avg_distance_err_img2 += self.get_point_2_line_error(pt1, line_2)
            epip_test_err += self.epipolar_test_single_point(pt1, pt2)

            # calculating 2 points on the line
            y_0 = self.epipoline(0, line_1)
            y_1 = self.epipoline(img_W, line_1)

            y_0 = self.epipoline(0, line_2)
            y_1 = self.epipoline(img_W, line_2)

            # Set color for line
            color = self.colors[color_idx % len(self.colors)]

            # drawing the line and feature points on the left image
            img1_line = cv2.circle(img1_line, (int(x1), int(y1)), radius=4, color=color)
            img1_line = cv2.line(img1_line, (0, y_0), (img_W, y_1), color=color, lineType=cv2.LINE_AA)
            # displaying just feature points
            cv2.circle(self.img1.copy(), (int(x1), int(y1)), radius=4, color=color)

            # drawing the line on the right image
            img2_line = cv2.circle(img2_line, (int(x2), int(y2)), radius=4, color=color)
            img2_line = cv2.line(img2_line, (0, y_0), (img_W, y_1), color=color, lineType=cv2.LINE_AA)
            # displaying just feature points
            cv2.circle(self.img2.copy(), (int(x2), int(y2)), radius=4, color=color)

        avg_distance_err_img1 /=  pts1.shape[0]
        avg_distance_err_img2 /=  pts1.shape[0]
        epip_test_err /= pts1.shape[0]

        vis = torch.cat((img1_line, img2_line), axis=0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_H = vis.shape[0]
        cv2.putText(vis, str(avg_distance_err_img1), (10, 20), font, 0.6, color=(128, 0, 0), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(avg_distance_err_img2), (10, img_H - 10), font, 0.6, color=(0, 0, 128), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(epip_test_err), (10, 210), font, 0.6, color=(130, 0, 150), lineType=cv2.LINE_AA)

        if(avg_distance_err_img1 > 13 or abs(epip_test_err) > 0.01):
            if move_bad_images:
                src_path1 = os.path.join(self.sequence_path, "image_0", self.file_name1)
                dst_path1 = os.path.join(self.sequence_path, "BadFrames", self.file_name1)
                if os.path.exists(src_path1):  
                    print(f'moved {src_path1} to {dst_path1}')
                    os.rename(src_path1, dst_path1)         
            else:                
                cv2.imwrite(os.path.join(sqResultDir, "bad_frames", 'epipoLine_sift_{}.png'.format(img_idx)), vis)
                print(os.path.join(sqResultDir, "bad_frames", 'epipoLine_sift_{}.png\n'.format(img_idx)))

        elif not move_bad_images:
            cv2.imwrite(os.path.join(sqResultDir, "good_frames", 'epipoLine_sift_{}.png'.format(img_idx)), vis)
            print(os.path.join(sqResultDir, "good_frames", 'epipoLine_sift_{}.png\n'.format(img_idx)))    
    
