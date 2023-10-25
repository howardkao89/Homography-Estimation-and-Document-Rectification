import sys
import numpy as np
import cv2 as cv
import math

np.seterr(invalid="ignore")


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])


def mouse_click(img):
    points_selected = []
    WINDOW_NAME = "Select Corners"
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_selected])

    while True:
        img_ = img.copy()
        for i, p in enumerate(points_selected):
            # draw points on img_
            cv.circle(img_, tuple(p), 5, (0, 255, 0), -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27:
            break  # exist when pressing ESC

    cv.destroyAllWindows()
    cv.waitKey(1)

    print("{} points selected ...".format(len(points_selected)))

    return points_selected


def DLT(anc_sam, tar_sam):
    A = []
    for i in range(4):
        u, v, u_pri, v_pri = anc_sam[i][0], anc_sam[i][1], tar_sam[i][0], tar_sam[i][1]
        A.append([0, 0, 0, -u, -v, -1, v_pri * u, v_pri * v, v_pri])
        A.append([u, v, 1, 0, 0, 0, -u_pri * u, -u_pri * v, -u_pri])

    U, S, Vh = np.linalg.svd(np.matrix(A))

    H = Vh[-1].reshape(3, 3)

    return H


def NDLT(anc_sam, tar_sam):
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

    H_hat = DLT(anc_sam_nor, tar_sam_nor)

    H = np.linalg.inv(T_pri).dot(H_hat).dot(T)

    return H


def bilinear_interpolation(x, y, img):
    x1 = math.floor(x)
    x2 = math.ceil(x)
    y1 = math.floor(y)
    y2 = math.ceil(y)

    interpolation = ((x2 - x) * (y2 - y) * img[x1][y1] +
                     (x - x1) * (y2 - y) * img[x2][y1] +
                     (x2 - x) * (y - y1) * img[x1][y2] +
                     (x - x1) * (y - y1) * img[x2][y2]) / ((x2 - x1) * (y2 - y1))

    return interpolation


def wraping(H, img, dim1, dim2):
    rectified_img = np.zeros((dim1, dim2, 3), np.uint8)
    for i in range(dim1):
        if i % math.floor(dim1 / 10) == 0:
            print(f"Progress {math.ceil(i / (dim1 / 100))}.0% ...")

        for j in range(dim2):
            [[x, y, z]] = np.matmul(H, np.matrix([i, j, 1]).transpose()).transpose().tolist()
            rectified_img[i][j] = bilinear_interpolation(x / z, y / z, img)

    return rectified_img


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[USAGE] python 2.py [IMAGE PATH]")
        sys.exit(1)

    img = cv.imread(sys.argv[1])
    dim1 = np.shape(img)[0]
    dim2 = np.shape(img)[1]

    print("Select the 4 corners of the captured image ...")
    print("Upper left -> Upper right -> Lower left -> Lower right")
    print("Press ESC when finished ...")
    corners = mouse_click(img)

    if len(corners) != 4:
        print("# of selected points != 4 ...")
        sys.exit(2)

    tar_corners = []
    for x, y in corners:
        tar_corners.append([y, x])
    anc_corners = np.array([[0, 0], [0, dim2 - 1], [dim1 - 1, 0], [dim1 - 1, dim2 - 1]])

    H = NDLT(anc_corners, tar_corners)

    rectified_img = wraping(H, img, dim1, dim2)

    cv.imshow("Result", rectified_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.waitKey(1)
