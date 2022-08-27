import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import glob
import os
nx = 9
ny = 6
dir = os.path.dirname(os.path.abspath(__file__))
# print(dir)
cal_images = glob.glob('{}/camera_cal/calibration*.jpg'.format(dir))
# print(images)
objpoints = []
imgpoints = []
lst_images = []
objp = np.zeros((ny*nx, 3), np.float32)
# create nx*ny points of xyz coordinates
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)


def cal_undistort(objpoints, imgpoints, image):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image.shape[0:2], None, None)
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    return(undist, corners)


def cal_transform(undist, corners):
    img_size = (undist.shape[1], undist.shape[0])
    # undistcopy = np.copy(undist) # always draw on copy
    # imgwithcorners = cv2.drawChessboardCorners(undistcopy, (nx,ny), corners, ret)
    src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    offset = 50
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                      [img_size[0]-offset, img_size[1]-offset],
                      [offset, img_size[1]-offset]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)
    return(warped)


def maincal(cal_images, objpoints, imgpoints, objp):
    for file in cal_images:
        img = cv2.imread(file)
        # cv2.imshow("original", img)
        # cv2.waitKey(0)
        lst_images.append(img)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray",gray)
        # cv2.waitKey(0)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        print(corners)
        if ret == True:
            imgpoints.append(corners)
            print(corners)
            objpoints.append(objp)
            print(objp)
        else:
            print("unable to find corners")
            continue
        undist, corners = cal_undistort(objpoints, imgpoints, img)
        warped = cal_transform(undist, corners)

        # cv2.imshow("undist", undist)
        # cv2.waitKey(0)
        # cv2.imshow("warped", warped)
        # cv2.waitKey(0)
        # if cv2.waitKey(0) == ord('q'):
        #     break
        #     cv2.destroyAllWindows()
    return(imgpoints, objpoints)


if __name__ == '__main__':
    maincal(cal_images, objpoints, imgpoints, objp)

# main()
# original / expected object points
# final point would be [8,5,0] since origin at 0,0,0 and have nx by ny
# for file in images:
#     img = cv2.imread(file)
#     lst_images.append(img)
#     # cv2.imshow("img", img)
#     # cv2.waitKey(0)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # cv2.imshow("gray",gray)
#     # cv2.waitKey(0)
#     ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
#     print(corners)
#     if ret==True:
#         imgpoints.append(corners)
#         print(corners)
#         objpoints.append(objp)
#         print(objp)
#         # imgcopy = np.copy(img)
#         # imgwithcorners = cv2.drawChessboardCorners(imgcopy, (nx,ny), corners, ret)
#         # cv2.imshow("img",img)
#         # cv2.waitKey(0)
#     else:
#         print("Unable to read corners")
#
# def cal_undistort(objpoints,imgpoints,lst_images):
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,lst_images[1].shape[0:2], None, None)
#     print(ret,mtx,dist,rvecs,tvecs)
#     offset = 100
#     for image in lst_images:
#         # cv2.imshow("ig",image)
#         # cv2.waitKey(0)
#         undist = cv2.undistort(image, mtx, dist, None, mtx)
#         # cv2.imshow("undistorted", dst)
#         #cv2.imshow('regular',image)
#         # cv2.waitKey(0)
#         grayy = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
#
#         # cv2.imshow("gray",grayy)
#         # cv2.waitKey(0)
#         ret, corners = cv2.findChessboardCorners(grayy, (nx,ny), None)
#         if ret==True:
#             img_size = (grayy.shape[1], grayy.shape[0])
#             undistcopy = np.copy(undist) # always draw on copy
#             imgwithcorners = cv2.drawChessboardCorners(undistcopy, (nx,ny), corners, ret)
#             src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
#             # outer 4 corners
#             print("srcpoints: ",src)
#
#             dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
#                                      [img_size[0]-offset, img_size[1]-offset],
#                                      [offset, img_size[1]-offset]])
#             # Given src and dst points, calculate the perspective transform matrix
#             M = cv2.getPerspectiveTransform(src, dst)
#             # Warp the image using OpenCV warpPerspective()
#             warped = cv2.warpPerspective(undist, M, img_size)
#             cv2.imshow("warped", warped)
#             cv2.waitKey(0)
#         else:
#             print("no")
#
# cal_undistort(objpoints,imgpoints,lst_images)
