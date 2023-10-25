import sys
import numpy as np
import cv2 as cv
import random
import math


def get_sift_correspondences(img1, img2):
    """
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    """
    # sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()  # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    return points1, points2, kp1, kp2, good_matches


def DLT(anc_sam, tar_sam, k):
    A = []
    for i in range(k):
        u, v, u_pri, v_pri = anc_sam[i][0], anc_sam[i][1], tar_sam[i][0], tar_sam[i][1]
        A.append([0, 0, 0, -u, -v, -1, v_pri * u, v_pri * v, v_pri])
        A.append([u, v, 1, 0, 0, 0, -u_pri * u, -u_pri * v, -u_pri])

    U, S, Vh = np.linalg.svd(np.matrix(A))

    H = Vh[-1].reshape(3, 3)

    return H


def NDLT(anc_sam, tar_sam, k):
    anc_sam_mean = np.mean(anc_sam, axis=0)
    anc_sam_std = np.std(anc_sam, axis=0)
    anc_sam_nor = (anc_sam[:] - anc_sam_mean) * math.sqrt(2) / anc_sam_std
    T = np.matrix([[math.sqrt(2) / anc_sam_std[0], 0, -anc_sam_mean[0] * math.sqrt(2) / anc_sam_std[0]],
                   [0, math.sqrt(2) / anc_sam_std[1], -anc_sam_mean[1] * math.sqrt(2) / anc_sam_std[1]],
                   [0, 0, 1]])

    tar_sam_mean = np.mean(tar_sam, axis=0)
    tar_sam_std = np.std(tar_sam, axis=0)
    tar_sam_nor = (tar_sam[:] - tar_sam_mean) * math.sqrt(2) / tar_sam_std
    T_pri = np.matrix([[math.sqrt(2) / tar_sam_std[0], 0, -tar_sam_mean[0] * math.sqrt(2) / tar_sam_std[0]],
                       [0, math.sqrt(2) / tar_sam_std[1], -tar_sam_mean[1] * math.sqrt(2) / tar_sam_std[1]],
                       [0, 0, 1]])

    H_hat = DLT(anc_sam_nor, tar_sam_nor, k)

    H = np.linalg.inv(T_pri).dot(H_hat).dot(T)

    return H


def compute_error(H, anc_gt, tar_gt, threshold):
    p_s = []
    for u, v in anc_gt:
        p_s.append([u, v, 1])

    p_t = []
    for u, v in tar_gt:
        p_t.append([u, v, 1])

    p_s = np.matrix(p_s)
    Hp_s = np.matmul(H, p_s.transpose()).transpose()

    p_t_hat = []
    for x, y, z in Hp_s.tolist():
        p_t_hat.append([x / z, y / z, 1])

    inliers = 0
    error = 0
    p_t = np.array(p_t)
    p_t_hat = np.array(p_t_hat)
    for i in range(p_t.shape[0]):
        norm = np.linalg.norm(p_t[i] - p_t_hat[i])
        error += norm
        if norm <= threshold:
            inliers += 1
    error /= p_t.shape[0]

    return inliers, error


def RANSAC(anc_cor, tar_cor, anc_gt, tar_gt, k, iter, threshold):
    max_inliers = 0

    for t in range(iter):
        if t % (iter / 10) == 0:
            print(f'Progress {t / (iter / 100)}% ...')

        idx = random.sample(range(anc_cor.shape[0]), k)
        anc_sam = np.array([anc_cor[i] for i in idx])
        tar_sam = np.array([tar_cor[i] for i in idx])

        DLT_H = DLT(anc_sam, tar_sam, k)
        DLT_inliers, DLT_error = compute_error(DLT_H, anc_gt, tar_gt, threshold)

        NDLT_H = NDLT(anc_sam, tar_sam, k)
        NDLT_inliers, NDLT_error = compute_error(NDLT_H, anc_gt, tar_gt, threshold)

        if max_inliers < NDLT_inliers:
            max_inliers = NDLT_inliers
            ideal_anc_sam = anc_sam
            ideal_tar_sam = tar_sam
            ideal_idx = idx
            ideal_DLT_H = DLT_H
            ideal_DLT_error = DLT_error
            ideal_NDLT_H = NDLT_H
            ideal_NDLT_error = NDLT_error

    print(f'Progress 100.0% ...')

    return ideal_anc_sam, ideal_tar_sam, ideal_idx, ideal_DLT_H, ideal_DLT_error, ideal_NDLT_H, ideal_NDLT_error


if __name__ == "__main__":
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])

    points1, points2, kp1, kp2, good_matches = get_sift_correspondences(img1, img2)

    tar_dim1 = np.shape(img2)[0]
    tar_dim2 = np.shape(img2)[1]

    iter = 50000
    threshold = math.log(math.sqrt(math.pow(tar_dim1, 2)+ math.pow(tar_dim2, 2))) * 0.0125

    for k in [4, 8, 20]:
        print(f"k = {k}")

        anc_sam, tar_sam, idx, DLT_H, DLT_error, NDLT_H, NDLT_error = RANSAC(points1, points2, gt_correspondences[0], gt_correspondences[1], k, iter, threshold)

        print(f"k = {k}")
        print("Anchor sample:")
        print(anc_sam)
        print("Target sample:")
        print(tar_sam)
        print("DLT H =")
        print(DLT_H)
        print(f"DLT error = {DLT_error}")
        print("NDLT H =")
        print(NDLT_H)
        print(f"NDLT error = {NDLT_error}", end='\n\n')

        matches = []
        for i in idx:
            matches.append(good_matches[i])
        img_draw_matches = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow(f'k = {k} matches', img_draw_matches)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.waitKey(1)
