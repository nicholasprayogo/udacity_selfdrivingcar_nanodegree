# **Traffic Sign Recognition** 

## Writeup

written in 2019
Template given by Udacity

---
[image1]: ./images/distrib.png
[image2]: ./images/bewareice.png
[image3]: ./images/speed30.png
[image4]: ./images/curv2right.jpg
[image5]: ./images/stop.jpg
[image6]: ./images/noover.jpg

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Please refer to project code at Traffic_Sign_Classifier_FINAL.html in this workspace. 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34995
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32,32
* The number of unique classes/labels in the data set is 43

#### 2. Include a visualization of the dataset.

![alt text][image1]
As seen here, the data is quite evenly distributed among the classes. However, there's indeed more data for particular classes that could potentially cause the model to lean towards certain classes, but using proper data augmentation, we could solve this issue.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I provided 2 modes of images, either keeping the images in rgb as supplied, or converting it to grayscale. I find that using my model architecture, classifying over rgb and grayscale images both perform equally well. 

In terms of preprocessing, I introduced equalize histogram and normalization as suggested by the previous project reviewer. Keeping the data between -1 and 1 helps the model converge faster, generalize better, and obtain better accuracy. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Initial LeNet model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6.  				|
| Convolution 5x5	    | out 10x10x16 valid padding        			|
| RELU           		|                   							|
| Max pooling     		| 2x2 stride out 5x5x16 			    		|
| FC 120         		| in 400 out 120     							|
| FC 84					|in 120 out 84									|
| FC 10					|	in 84 out 10								|

Final model: 
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6.  				|
| Convolution 5x5	    | out 10x10x16 valid padding        			|
| ELU           		|                   							|
| Max pooling     		| 2x2 stride out 5x5x16 			    		|
| FC 120   +dropout   	| in 400 out 120     							|
| FC 84	    +dropout	|in 120 out 84									|
| FC 10	        		|	in 84 out 10								|

In terms of layer sizes, I simply used the formulas given in the course materials to ensure the propagation of the input is feasible. Explanation about how I came up with the final model is written below.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Loss: cross entropy, because it is proven to be accurate in gauging loss for classification tasks across discrete space. 
Optimizer: Adam, because it has been proven to be superior among other conventional optimizers such as SGD and Adagrad. However, for this use case, SGD would work just as fine with not a lot of difference in terms of performance and convergence time.
learning rate: 0.001 (standard practice)
batch size: 64. Using a larger batch size compared to say 32 or 64 helps in terms of introducing useful noise that could prevent the model from overfitting. However, since using different batch sizes did not really affect the performance significantly, I chose 64 which converges to high validation accuracy slightly faster than 128.  
epochs: 30 (using the final model, 100 for initial model). 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.927 (rounded off to 0.93 which satisfies the requirements)

If a well known architecture was chosen:
* What architecture was chosen? 
LeNet architecture was chosen because it worked quite well, only certain hyperparameters such as the size of the convolutional and FC layers that need to be tweaked in order for the architecture to work with the dataset.
* Why did you believe it would be relevant to the traffic sign application?
Traffic sign classification is a simple classification problem with a discrete classification space. LeNet has been proven to work well for such cases and thus it was implemented.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
With 93% accuracy on the validation set, this out-of-the-box LeNet-based model works well.
 
Iterative process:

Using the initial LeNet model, the model was able to reach 0.927 validation accuracy in 100 epochs. While this is a great result considering I did not tweak the legacy architecture very much, as suggested by the reviewer, I experimented with the model to see if it can converge faster and obtain higher validation accuracy. 

I added dropout layers after the convolutional layers, but this did not help a lot considering the max pooling layers already contribute to reducing overfitting. 

Hence, I moved the dropout layers to after the fully connected layers. With a keep probability of 0.5, the model converges to around 0.93 accuracy at 30 epochs. This was significantly faster than the initial model, and it proves how the dropout layers aid in faster convergence and validation accuracy.

Furthermore, I resorted to using Elu instead of ReLu activations, seeking to improve convergence rate and validation accuracy. Sure enough, the model performed even better, achieving 0.939 validation accuracy at 14 epochs, and 0.945 at 30 epochs. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

![alt text][image2]

This beware of ice/snow sign could be difficult to classify because at lower resolution the sign could be misleading.

![alt text][image3]

![alt text][image4]

This can also be misleading as double curve due to its orientation. Hence, proper data augmentation might be required.

![alt text][image5]
![alt text][image6]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roadwork     		    | Roadwork  									| 
| Speedlimit 70     	| Speedlimit 70 					    		|
| No cars over 3.5 tons	| Yield											|
| Bumpy road      		| Bumpy Road					 				|
| Stop       			| Stop               							|
| Curve to right    	| Double curve      							|
| no vehicles       	| no vehicles        							|
| beware ice          	| beware ice         							|
| speed limit 30      	| speed limit 30	                        	|


The model was able to correctly guess 7 of the 9 traffic signs, which gives an accuracy of 78%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For most instances, the model was highly confident (over 99%) over its prediction as mirrored by the softmax probabilities. 

```
Image: 0, Truth: 25
Pred 1: 25, confidence: 0.9998564720153809
Pred 1: 4, confidence: 0.00014353101141750813
Pred 1: 27, confidence: 9.459156125046775e-09
Pred 1: 31, confidence: 2.932172950623202e-12
Pred 1: 5, confidence: 1.9407506746343698e-17
Image: 1, Truth: 4
Pred 1: 4, confidence: 1.0
Pred 1: 0, confidence: 0.0
Pred 1: 18, confidence: 0.0
Pred 1: 40, confidence: 0.0
Pred 1: 11, confidence: 0.0
Image: 2, Truth: 10
Pred 1: 12, confidence: 1.0
Pred 1: 16, confidence: 0.0
Pred 1: 40, confidence: 0.0
Pred 1: 9, confidence: 0.0
Pred 1: 41, confidence: 0.0
Image: 3, Truth: 22
Pred 1: 22, confidence: 1.0
Pred 1: 15, confidence: 0.0
Pred 1: 29, confidence: 0.0
Pred 1: 13, confidence: 0.0
Pred 1: 25, confidence: 0.0
Image: 4, Truth: 14
Pred 1: 14, confidence: 1.0
Pred 1: 1, confidence: 5.887566035625171e-37
Pred 1: 5, confidence: 0.0
Pred 1: 34, confidence: 0.0
Pred 1: 13, confidence: 0.0
Image: 5, Truth: 20
Pred 1: 26, confidence: 1.0
Pred 1: 39, confidence: 1.8043601721529513e-15
Pred 1: 12, confidence: 1.7789441977103096e-21
Pred 1: 30, confidence: 9.402225067265488e-25
Pred 1: 25, confidence: 8.551162479667799e-25
Image: 6, Truth: 15
Pred 1: 15, confidence: 1.0
Pred 1: 40, confidence: 0.0
Pred 1: 25, confidence: 0.0
Pred 1: 12, confidence: 0.0
Pred 1: 1, confidence: 0.0
Image: 7, Truth: 30
Pred 1: 30, confidence: 1.0
Pred 1: 11, confidence: 5.387215002021899e-30
Pred 1: 21, confidence: 2.3868968214527311e-32
Pred 1: 23, confidence: 0.0
Pred 1: 10, confidence: 0.0
Image: 8, Truth: 1
Pred 1: 1, confidence: 1.0
Pred 1: 2, confidence: 0.0
Pred 1: 4, confidence: 0.0
Pred 1: 0, confidence: 0.0
Pred 1: 5, confidence: 0.0
```


