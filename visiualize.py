from params import *
from FunMatrix import *
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

    # @staticmethod
    # def convertP(pose1, pose2):
    #     R1, T1 = pose1
    #     R2, T2 = pose2
    #     # return R2, T2
    #     newR = np.dot(np.linalg.inv(R2), R1)
    #     newT = np.dot(np.dot(np.linalg.inv(R1), R2), T2) - T1
    #     #
    #     # newR = np.dot(R2, np.linalg.inv(R1))
    #     # newT = np.dot(R1, T1-T2)
    #     return newR, newT

    def EMat(self, R, T):
        # print(T)
        t = T
        T = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ], dtype=float)


        E = T.dot(R)
        return E

    def visualize(self, sqResultDir, img_idx, K, THRESHOLD=0.15):
        left_img = cv2.imread(self.leftImg)
        img1 = cv2.cvtColor(left_img.copy(), cv2.COLOR_BGR2GRAY)
        left_img_line = img1.copy()

        right_img = cv2.imread(self.rightImg)
        img2 = cv2.cvtColor(right_img.copy(), cv2.COLOR_BGR2GRAY)
        right_img_line = img2.copy()

        # f_mat = self.EMat(R=self.R, T=self.T)
        E = compute_essential(self.R, self.T)
        f_mat = compute_fundamental(E, K, K)    

        sift = cv2.xfeatures2d.SIFT_create()
        bf = cv2.BFMatcher()

        (kp1, des1) = sift.detectAndCompute(img1, None)
        (kp2, des2) = sift.detectAndCompute(img2, None)

        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < THRESHOLD * n.distance:
                good.append(m)

        # drawing epipolar line
        err_l = []
        err_r = []
        img_W = left_img.shape[1] - 1
        for color_idx, m in enumerate(good):
            # get the feature points in both left and right images
            x_l, y_l = kp1[m.queryIdx].pt
            x_r, y_r = kp2[m.trainIdx].pt

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

            print("Point {}: ".format(color_idx), self.verify_xFx(point_l, f_mat, point_r))

            # drawing the line on the right image
            right_img_line = cv2.circle(right_img_line, (int(x_r), int(y_r)), radius=4, color=color)
            right_img_line = cv2.line(right_img_line, (0, y_0), (img_W, y_1), color=color, lineType=cv2.LINE_AA)
            # displaying just feature points
            right_img = cv2.circle(right_img, (int(x_r), int(y_r)), radius=4, color=color)

        l_avgErr = np.average(err_l) if err_l else 0
        r_avgErr = np.average(err_r) if err_r else 0

        shape = left_img.shape
        vis = np.concatenate((left_img_line, right_img_line), axis=0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_H = vis.shape[0]
        cv2.putText(vis, str(l_avgErr), (10, 20), font, 0.5, color=(0, 255, 0), lineType=cv2.LINE_AA)
        cv2.putText(vis, str(r_avgErr), (10, img_H - 10), font, 0.5, color=(0, 255, 0), lineType=cv2.LINE_AA)

        cv2.imwrite(os.path.join(sqResultDir, 'epipoLine_sift_{}.png'.format(img_idx)), vis)
        print(os.path.join(sqResultDir, 'epipoLine_sift_{}.png'.format(img_idx)))

    @staticmethod
    def verify_xFx(point1, F, point2):
        return point2.T.dot(F).dot(point1)

    @staticmethod
    def verify_xfx(point, l):
        threshold = 2
        l = l.flatten()
        # K = EpiLine.d['P0'][0:3, 0:3]
        result = abs(np.dot(point, l.T) / np.sqrt(l[0] * l[0] + l[1] * l[1]))

        if result <= threshold:
            # print(True, result)
            return (True, result)
        # print(False, result)
        return (False, result)

# left_projection_matrix = process_calib('sequences\\02\\calib.txt')

# K, _ = get_internal_param_matrix(left_projection_matrix)

# poses = read_poses('poses\\02.txt')

# for i in range(len(poses) - 1):
#     R_relative, t_relative = compute_relative_transformations(poses[i], poses[i+jump_frames])

#     lImg = f'sequences\\02\\image_0\\{i:06}.png'
#     rImg = f'sequences\\02\\image_0\\{i+jump_frames:06}.png'

#     a = EpipoLine(leftImg=lImg, rightImg=rImg, R=R_relative, T=t_relative)

#     a.visualize(sqResultDir='epipoles', img_idx=i, K=K, THRESHOLD=0.2)

