from params import *
from FunMatrix import *
from utils import *
from Dataset import train_loader, val_loader
import os
import numpy as np
import cv2

colors = [
        (255, 102, 102),
        (102, 255, 255),
        (125, 125, 125),
        (204, 229, 255),
        (0, 0, 204)
    ]
class EpipoLine:
    def __init__(self, leftImg, rightImg, R, T):
        self.leftImg = leftImg
        self.rightImg = rightImg
        self.R = R
        self.T = T

    @staticmethod
    def epipoline(x, formula):
        array = formula.flatten()
        a = array[0]
        b = array[1]
        c = array[2]
        return int((-c - a * x) / b)


    def visualize(self, sqResultDir, img_idx, K, THRESHOLD=0.15):
        sift = cv2.xfeatures2d.SIFT_create()
        bf = cv2.BFMatcher()

        # f_mat = self.FMat(R=self.R, T=self.T)
        E = compute_essential(self.R, self.T)
        f_mat = compute_fundamental(E, K, K)  

        left_img = cv2.imread(self.leftImg)
        left_imgG = cv2.cvtColor(left_img.copy(), cv2.COLOR_BGR2GRAY)
        left_img_line = left_img.copy()

        right_img = cv2.imread(self.rightImg)
        right_imgG = cv2.cvtColor(right_img.copy(), cv2.COLOR_BGR2GRAY)
        right_img_line = right_img.copy()

        (kps_left, descs_left) = sift.detectAndCompute(left_imgG, None)
        (kps_right, descs_right) = sift.detectAndCompute(right_imgG, None)

        matches = bf.knnMatch(descs_left, descs_right, k=2)
        good = []
        for m, n in matches:
            if m.distance < THRESHOLD * n.distance:
                good.append(m)

        # drawing epipolar line
        err_l = []
        err_r = []
        img_W = left_img.shape[1] - 1
        xfx = 0
        for color_idx, m in enumerate(good):
            # get the feature points in both left and right images
            x_l, y_l = kps_left[m.queryIdx].pt
            x_r, y_r = kps_right[m.trainIdx].pt

            '''Color for line'''
            color = colors[color_idx % len(colors)]

            '''Epi line on the left image'''
            # epi line of right points on the left image
            point_r = np.array([x_r, y_r, 1])

            line_l = np.dot(f_mat.T, point_r)
            # verifying points
            _, err_L = self.verify_xfx(point_r, line_l)
            err_l.append(err_L)
            # calculating 2 points on the line
            y_0 = self.epipoline(0, line_l)
            y_1 = self.epipoline(img_W, line_l)
            # drawing the line and feature points on the left image
            left_img_line = cv2.circle(left_img_line, (int(x_l), int(y_l)), radius=4, color=color)
            left_img_line = cv2.line(left_img_line, (0, y_0), (img_W, y_1), color=color, lineType=cv2.LINE_AA)
            # displaying just feature points
            left_img = cv2.circle(left_img, (int(x_l), int(y_l)), radius=4, color=color)

            '''Epi line on the right image'''
            # epi line of left points on the right image
            point_l = np.array([x_l, y_l, 1])
            line_r = np.dot(f_mat, point_l)

            # verifying points
            _, err_R = self.verify_xfx(point_l, line_r)
            err_r.append(err_R)
            # calculating 2 points on the line
            y_0 = self.epipoline(0, line_r)
            y_1 = self.epipoline(img_W, line_r)

            xfx += self.verify_xFx(point_l, f_mat, point_r)

            # drawing the line on the right image
            right_img_line = cv2.circle(right_img_line, (int(x_r), int(y_r)), radius=4, color=color)
            right_img_line = cv2.line(right_img_line, (0, y_0), (img_W, y_1), color=color, lineType=cv2.LINE_AA)
            # displaying just feature points
            right_img = cv2.circle(right_img, (int(x_r), int(y_r)), radius=4, color=color)
        
        if(len(good) > 0):
            xfx /= len(good)
        print(f'xFx: {xfx}')

        l_avgErr = np.average(err_l) if err_l else 0
        r_avgErr = np.average(err_r) if err_r else 0

        vis = np.concatenate((left_img_line, right_img_line), axis=0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_H = vis.shape[0]
        cv2.putText(vis, str(l_avgErr), (10, 20), font, 0.6, color=(128, 0, 0), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(r_avgErr), (10, img_H - 10), font, 0.6, color=(128, 0, 0), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(xfx), (10, 360), font, 0.6, color=(130, 0, 150), lineType=cv2.LINE_AA)

        if(l_avgErr > 7 or np.abs(xfx) > 0.1 or xfx == 0):
            cv2.imwrite(os.path.join(sqResultDir, "bad_frames", 'epipoLine_sift_{}.png'.format(img_idx)), vis)
            print(os.path.join(sqResultDir, "bad_frames", 'epipoLine_sift_{}.png\n'.format(img_idx)))
        else:
            cv2.imwrite(os.path.join(sqResultDir, "good_frames", 'epipoLine_sift_{}.png'.format(img_idx)), vis)
            print(os.path.join(sqResultDir, "good_frames", 'epipoLine_sift_{}.png\n'.format(img_idx)))

    @staticmethod
    def verify_xFx(point1, F, point2):
        return point2.T.dot(F).dot(point1)


    @staticmethod
    def verify_xfx(point, l):
        threshold = 2
        l = l.flatten()
        result = abs(np.dot(point, l.T) / np.sqrt(l[0] * l[0] + l[1] * l[1]))

        if result <= threshold:
            return (True, result)
        return (False, result)

bad_frames_dir = os.path.join('epipole_lines', "bad_frames")
good_frames_dir = os.path.join('epipole_lines', "good_frames")

os.makedirs(bad_frames_dir, exist_ok=True)
os.makedirs(good_frames_dir, exist_ok=True)

left_projection_matrix = read_calib('sequences\\02\\calib.txt')

K, _ = get_intrinsic(left_projection_matrix)

poses = read_poses('poses\\02.txt')

for i in range(len(poses) - 1):
    R_relative, t_relative = compute_relative_transformations(poses[i], poses[i+jump_frames])

    lImg = f'sequences\\02\\image_0\\{i:06}.png'
    rImg = f'sequences\\02\\image_0\\{i+jump_frames:06}.png'

    a = EpipoLine(leftImg=lImg, rightImg=rImg, R=R_relative, T=t_relative)

    a.visualize(sqResultDir='epipole_lines', img_idx=i, K=K, THRESHOLD=0.15)


