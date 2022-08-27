import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import glob
import os

from thresholding_utils import abs_sobel_thresh, mag_thresh, dir_threshold, hls_select
from misc_utils import show_img, showstep, adjust_original_image, draw_on_original, mask, transform
from lanedetect import sliding_window, fit_polynomial, fit_poly, search_around_poly
from calibration import cal_transform, cal_undistort, maincal

dir_path = os.path.dirname(os.path.abspath(__file__))
nx = 9
ny = 6
SCALAR_EBLUE = (255, 255, 102)

cal_images = glob.glob('./camera_cal/calibration*.jpg')


objpoints = []
imgpoints = []
lst_images = []
objp = np.zeros((ny*nx, 3), np.float32)
# create nx*ny points of xyz coordinates
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

ym_per_pix = 30 / 720   # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


def measure_curvature_pixels(ploty, left_fit, right_fit, left_fitx, right_fitx):
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    yval = np.max(ploty)

    A = left_fit[0]
    B = left_fit[1]
    C = left_fit[2]

    d1_left = 2*A*yval + B
    d2_left = 2*A
    left_curverad = (1+(d1_left)**2)**(3/2)/abs(d2_left) * ym_per_pix

    D = right_fit[0]
    E = right_fit[1]
    F = right_fit[2]

    d1_right = 2*D*yval + E
    d2_right = 2*D
    right_curverad = (1+(d1_right)**2)**(3/2)/abs(d2_right) * ym_per_pix

    lane_middle = left_fitx[0] + (right_fitx[0] - left_fitx[0])/2.0

    deviation = (lane_middle - 640)*xm_per_pix
    return left_curverad, right_curverad, deviation


def main():
    # calibration
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    imgpointsa, objpointsa = maincal(cal_images, objpoints, imgpoints, objp)

    dir_vid = dir_path + "/test_videos"
    print(dir_vid)
    vid = dir_vid + "/project_video.mp4"

    cap = cv2.VideoCapture(vid)
    print(cap.isOpened())
    out = cv2.VideoWriter('./test_out/pipeline.mp4', fourcc, 30.0, (1280, 695))
    while cap.isOpened():
        ret, img = cap.read()
        img, corners = cal_undistort(objpointsa, imgpointsa, img)
        # cv2.imshow("original", img)
        gradbinary = abs_sobel_thresh(img, thresh_min=10, thresh_max=255)
        magbinary = mag_thresh(img, mag_thresh=(10, 255))
        # experiment with the combination of threshold values
        dirbinary = dir_threshold(img, sobel_kernel=3, thresh=(0.5, 1.3))
        hlsbinary = hls_select(img, thresh=(70, 255), thresh2=(0, 255))
        # show_img(hlsbinary, showstep=False, name="hls")

        combined = np.zeros_like(dirbinary)
        #combined[(gradbinary == 255)&((magbinary == 255) & (dirbinary == 255))] = 255
        combined[(hlsbinary == 255) & (magbinary == 255)] = 255
        height_adjustment = 25
        vertices, masked_image, img2 = mask(combined, adjustment=height_adjustment)

        imgOriginalAdjusted = adjust_original_image(img, adjustment=height_adjustment)

        # cv2.imshow("adjusted", imgOriginalAdjusted)
        # print(masked_image)
        # show_img(masked_image, showstep=True, name="masked")
        # cv2.imshow("masked", masked_image)
        warped = transform(masked_image, vertices)

        histogram, leftx, lefty, rightx, righty, out_img = sliding_window(warped)

        out_img, left_fit, right_fit, ploty = fit_polynomial(warped)
        result, left_fitx, right_fitx, pts = search_around_poly(warped, left_fit, right_fit)
        # plt.plot(histogram)
        # plt.show()
        # plt.pause(0.0005)
        # show_img(out_img, name="sliding window", showstep=True)
        # show_img(result, name="polyfit", showstep=True)

        dewarped = transform(result, vertices, mode="dewarp")
        # show_img(dewarped, name="dewarped", showstep=True)

        blended = draw_on_original(imgOriginalAdjusted, leftx, lefty, rightx, righty,
                                   vertices, pts)

        left_curverad, right_curverad, deviation = measure_curvature_pixels(
            ploty, left_fit, right_fit, left_fitx, right_fitx)
        # print("radius: ", left_curverad, right_curverad)

        cv2.putText(blended, "Left curvature: {}, Right curvature: {}, Deviaton: {}".format(
            left_curverad, right_curverad, deviation), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_EBLUE, 2)

        # print(blended.shape)
        # show_img(blended, name="blended", showstep=True)
        out.write(blended)
        if cv2.waitKey(1) == ord("q"):
            break
            cap.release()
            cv2.destroyAllWindows()


# plt.ion()
main()

# use this to debug
# for i in range(20,150,5):
#     abs_sobel_thresh(img, thresh_min=i, thresh_max=255)

# for i in range(5,150,5):
#     mag_thresh(img, sobel_kernel=3, mag_thresh=(i, 255))
#     print(i)
# at 50 start to get less noisier lane lines

# for i in range(10,150,5):
#     hls_binary = hls_select(img, thresh=(i, 255))

# mag_thresh(img)
# dir_threshold(img)
