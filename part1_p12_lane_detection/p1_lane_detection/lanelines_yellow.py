import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math
# import threading

showstep = False
# SET TO TRUE IF WANT TO SHOW STEPS

rho = 1
theta = np.pi/180
threshold = 70
min_line_len = 100
max_line_gap = 160

dir_path = os.path.dirname(os.path.realpath(__file__))

def showfig(image, colormap):
    print("start")
    plt.imshow(image, cmap=colormap)
    plt.show()


def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if showstep:
        cv2.imshow("gray", gray)
        cv2.waitKey(0)
    return gray


def gaussian_blur(img, kernel_size):
    blur = cv2. GaussianBlur(img, (kernel_size, kernel_size), 0)
    if showstep:
        cv2.imshow("blur", blur)
        cv2.waitKey(0)
    return blur


def canny(img, low_threshold, high_threshold):
    can = cv2.Canny(img, low_threshold, high_threshold)
    if showstep:
        cv2.imshow("Can", can)
        cv2.waitKey(0)

    # t = threading.Thread(target=showfig, name="canny", args=(can, 'gray'))
    # t.start()
    # plt.show()

    return can


def roi(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
        print(ignore_mask_color)
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    if showstep:
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    if showstep:
        cv2.imshow("mask", masked_image)
        cv2.waitKey(0)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=4):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, imgOriginal, rho, theta, threshold, min_line_len, max_line_gap, vertices):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    # # Testing for optimal threshold value
    # for i in range(50, threshold, 5):
    #     print(i)
    #     lines = cv2.HoughLinesP(img, rho, theta, i, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #     draw_lines(line_img, lines)
    #     plt.imshow(line_img, cmap='gray')
    #     plt.show()

    # print(i)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    # line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    shape = img.shape
    width = shape[1]
    height = shape[0]

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)

    leftlines = []
    rightlines = []

    vertices_left = [vertices[0][0][0], vertices[0][0][1]]
    vertices_right = [vertices[0][3][0], vertices[0][3][1]]

    for line in lines:
        # print(line)
        if line[0][2] < width/2:
            leftlines.append(line)
        else:
            rightlines.append(line)

    # select only 1 line for left and right each
    if len(leftlines) != 0:
        left_line = np.mean(leftlines, axis=0)
        left_line = left_line.astype(int)
        left_line1 = left_line[0]
        slope_left = (left_line1[3]-left_line1[1])/(left_line1[2]-left_line1[0])
        # print(slope_left)
        # extrapolate to the end of the ROI
        # note: here botleft and topleft is x[0],x[1], x[2],x[3]
        left_bot_x= int(left_line1[0] + (vertices_left[1]-left_line1[1])/slope_left)
        left_bot_y = vertices_left[1]

        left_line[0][0] = left_bot_x
        left_line[0][1] = left_bot_y

    else:
        left_line = []
        slope_left = 0

    if len(rightlines) != 0:
        right_line = np.mean(rightlines, axis=0)
        right_line = right_line.astype(int)
        right_line1 = right_line[0]
        slope_right = (right_line1[3]-right_line1[1])/(right_line1[2]-right_line1[0])
        # print(slope_right)
        # extrapolate to the end of the ROI
        # note: now top_right and bot_right is x[0],x[1], x[2],x[3]
        right_bot_x = int(right_line1[2] + (vertices_right[1]-right_line1[3])/slope_right)
        right_bot_y = vertices_right[1]

        right_line[0][2] = right_bot_x
        right_line[0][3] = right_bot_y
    else:
        right_line = []
        slope_right = 0

    lines = [left_line, right_line]
    print(lines)


    line_img = np.copy(imgOriginal)
    draw_lines(line_img, lines)
    # matplotlib is not thread friendly
    # t = threading.Thread(target=showfig, name="hough {}".format(i), args=(img, None))
    # t.start()
    if showstep:
        cv2.imshow("lines", line_img)
        cv2.waitKey(0)

    return line_img


def weighted_img(img, initial_img, α=0.5, β=0.5, γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)


def main():
    dir_vid = dir_path + "/test_videos"
    vid = dir_vid + "/solidYellowLeft.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture(vid)
    ret, imgOriginal = cap.read()
    height = imgOriginal.shape[0]
    width = imgOriginal.shape[1]

    c1 = (100, height) # x1, y1
    c2 = (width/2 - 100, height/2 + 50) # x2, y2
    c3 = (width/2 + 100, height/2 + 50) # x2 , y2
    c4 = (width-100, height) # x1, y1
    vertices = np.array([[c1, c2, c3, c4]], dtype=np.int32)
    print(vertices)
    # important to make it array of array, and dtype 32 int, because will need for polyfit

    out = cv2.VideoWriter(dir_path + '/test_videos_output/solidYellowOut.mp4',
                          fourcc, 30.0, (width, height))

    while cap.isOpened():
        ret, imgOriginal = cap.read()
        img = canny(gaussian_blur(grayscale(imgOriginal), 5), 50, 150)
        masked = roi(img, vertices)
        line_img = hough_lines(masked, imgOriginal, rho, theta,
                               threshold, min_line_len, max_line_gap, vertices)
        weighted = weighted_img(line_img, imgOriginal)
        cv2.imshow("Weighted Image", weighted)
        out.write(weighted)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


main()
