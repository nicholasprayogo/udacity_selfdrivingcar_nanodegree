# Projects: Lane Detection

## Project 1 (mini): Basic Lane Detection

Steps:
* Selecting color spaces (chosen grayscale for this)
* Masking the image only on the region of interest using `cv2.fillPoly` and `cv2.bitwise_and`
* Gaussian smoothing/blurring 
* Canny edge detection to thin out edges by using only the strongest gradients
* Hough transform: Finding the intersection of lines in Hough space to detect a line in original image space.

## Project 2: Advanced Lane Detection

For this specific project, back in 2019 when I did the project, I followed a rubric given by Udacity to write the description of the project.

Steps:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./p2_adv_lane_detection/images/dist.png
[image2]:  ./p2_adv_lane_detection/images/undist.png
[image3]: ./p2_adv_lane_detection/images/undist_road.png
[image4]: ./p2_adv_lane_detection/images/adjusted.png
[image5]: ./p2_adv_lane_detection/images/masked.png
[image6]: ./p2_adv_lane_detection/images/warpedroad.png
[image7]: ./p2_adv_lane_detection/images/slidewin.png
[image8]: ./p2_adv_lane_detection/images/polyfit.png
[image9]: ./p2_adv_lane_detection/images/dewarpedlane.png
[image10]: ./p2_adv_lane_detection/images/final.png
[video1]: ./p2_adv_lane_detection/pipeline.mp4

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Code: calibration.py

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

Original image:
![alt text][image1]

Undistorted:
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the object and image points from the camera calibration step, I was able to undistort the images of the highway road.

![alt text][image3]

I also cropped irrelevant parts of the image/frame to smoothen results (function adjust_original_image).

![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

As shown in main.py line 81, after experimentation (as can be seen from the commented code on thresholding_utils.py), I've decided to use the combination of the HLS and gradient magnitude binaries to generated the thresholded binary image. Then, I simply masked the image according to the region of interest. For the code I've used for masking, refer to misc_utils.py at the mask function.

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

By identifying the source and destination vertices, I could simply utilize `cv2.getPerspectiveTransform` to obtain a warp matrix, or using the `cv2.WARP_INVERSE_MAP` flag for the case of dewarping (useful for plotting the lane lines back into the original image).

```python
img_size = (undist.shape[1], undist.shape[0])
src = np.float32(vertices)
width1 = np.sqrt((topright[0]-topleft[0])**2+(topright[1]-topleft[1])**2)
width2 = np.sqrt((botright[0]-botleft[0])**2+(botright[1]-botleft[1])**2)
max_width = max(int(width1), int(width2))

height1 = np.sqrt((topright[0]-botright[0])**2+(topright[1]-botright[1])**2)
height2 = np.sqrt((botright[0]-topright[0])**2+(botleft[1]-topleft[1])**2)
max_height = max(int(height1), int(height2))

dst = np.float32([[0, max_height], [0, 0], [max_width, 0], [max_width, max_height]])

M = cv2.getPerspectiveTransform(src, dst) # or cv2.WARP_INVERSE_MAP for dewarp
```

This was the resultant warped image.
![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the sliding window approach (where it searches for nonzero pixels from the binary image) and fitting the lane line positions with a 2nd order polynomial.

##### Sliding window
![alt text][image7]

##### Polynomial fit
![alt text][image8]

##### Dewarped
![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `main.py` in the function  `measure_curvature_pixels`.

What the code does is finding the 2nd order coefficients to the left and right lines separately (based on the pixel posiitons obtained in the previous step) to measure the curvature, and finding the midpoint between the 2 lines and finding its deviation from the frame's center. If the deviation is negative it means the car is on the left of the center and vice versa.

It is also important to note that the straighter the line, the curvature approaches infinity.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Plotting the identified lane area back to the original image, as well as displaying the lane line curvatures and deviation in meters.

![alt text][image10]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Please refer to `./test_out/pipeline.mp4` for the full video of the pipeline.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

While it was a really fun project to do, there were indeed difficulties throughout. For instance, figuring out the vertices for perspective transformations was a challenge but a bit of experimentation solved this hurdle. Also, plotting the lane line pixels as well as coloring the lane regions were indeed challenging yet interesting tasks.

Some possible failure modes could be when the weather or lighting is unfriendly (e.g. dark/rainy). This could be mitigated if a camera with good response towards change in lighting were to be used since the picture's brightness could be adjusted accordingly. Another way this pipeline could fail is if the upcoming lane lines could not be seen (e.g. doing a steep turn in a narrow wiggly road). This requires the pipeline to be able to predict the curvature /trajectory of the lane lines ahead of time, perhaps using neural networks or reinforcement learning. It is interesting to see how Tesla has implemented this in there autopilot fleets, am curious to see how it's done and hopefully implement it in the future.

