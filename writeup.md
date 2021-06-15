**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
### Files Submitted & Code Quality

#####Architecture image and preprocessing visualization images are in [images] folder.

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
 1.model.py containing the script to create and train the model
 2.drive.py for driving the car in autonomous mode
 3.model.h5 containing a trained convolution neural network 
 4.writeup_report.md or writeup_report.pdf summarizing the results
#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy
###Model Overview:

I decided to test the model provided by NVIDIA as suggested by Udacity. The model architecture is described by NVIDIA 
here. As an input this model takes in image of the shape (60,266,3) but our dashboard
 images/training images are of size (160,320,3). I decided to keep the architecture of 
the remaining model same but instead feed an image of different input shape.
###Loading Data
I used the the dataset provided by Udacity
I am using OpenCV to load the images, by default the images are read by OpenCV in BGR
 format but we need to convert to RGB as in drive.py it is processed in RGB format.
Since we have a steering angle associated with three images we introduce a correction
 factor for left and right images since the steering angle is captured by the center angle.
I decided to introduce a correction factor of 0.2
For the left images I increase the steering angle by 0.2 and for the right images I 
decrease the steering angle by 0.2
###Preprocessing
I decided to shuffle the images so that the order in which images comes doesn't matters 
to the CNN
Augmenting the data- i decided to flip the image horizontally and adjust steering angle
 accordingly, I used cv2 to flip the images.
In augmenting after flipping multiply the steering angle by a factor of -1 to get the 
steering angle for the flipped image.
So according to this approach we were able to generate 6 images corresponding to one
 entry in .csv file
###Creation of the Training Set & Validation Set
I analyzed the Udacity Dataset and found out that it contains 9 laps of track 1 with
 recovery data. I was satisfied with the data and decided to move on.
I decided to split the dataset into training and validation set using sklearn 
preprocessing library.
I decided to keep 15% of the data in Validation Set and remaining in Training Set
I am using generator to generate the data so as to avoid loading all the images in the
 memory and instead generate it at the run time in batches of 32. Even Augmented images 
are generated inside the generators.
###Final Model Architecture
I made a little changes to the original NVIDIA architecture, my final architecture looks like in the image below

Layer              |output_shape           |param     
1.lambda_3         |(none,160,320,3)       |0
2.cropping2d_3     |(none,65,320,3)        |0
3.conv2d_1 (Conv2d)|(none,31,158,24)     |1824
4.activation(elu)  |(none,31,158,24)       |0
5.conv2d_2(Conv2D) |(none,14,77,36)        |21636
6.activation(elu)  |(none,14,77,36)        |0
7.conv2d_3(Conv2D) |(none,5,37,48)       |43248
8.activation(elu)  |(none,5,37,48)         |0
9.conv2d_4(Conv2D) |(none,3,35,64)         |27712
10.activation(elu) |(none,3,35,64)         |0
11.conv2d_5(Conv2D)|(none,1,33,64)         |36928
12.acrtivation(elu)|(none,1,33,64)         |0
13flatten          |(none,2112)            |0
14dense_1(Dense)   |(none,100)             |211300
15activation       |(none,100)             |0
16dropout          |(none,100)             |0
17dense_2(Dense)   |(none,50)              |5050
18.activation      |(none,50)              |0
19.dense_3(Dense)  |(none,10)              |510
20.actiavtion      |(none,10)              |0
21.Dense_4(Dense)  |(none,1)               |11

Total Params:348219

As it is clear from the model summary my first step is to apply normalization to the all the
 images.
Second step is to crop the image 70 pixels from top and 25 pixels from bottom. The image 
was cropped from top because 
I did not wanted to distract the model with trees and sky and 25 pixels from the bottom 
so as to remove the dashboard that is coming in the images
Next Step is to define the first convolutional layer with filter depth as 24 and filter size as (5,5) with (2,2) stride followed by ELU activation function
Moving on to the second convolutional layer with filter depth as 36 and filter size as (5,5) with (2,2) stride followed by ELU activation function
The third convolutional layer with filter depth as 48 and filter size as (5,5) with (2,2) stride followed by ELU activation function
Next we define two convolutional layer with filter depth as 64 and filter size as (3,3) and (1,1) stride followed by ELU activation funciton
Next step is to flatten the output from 2D to side by side
Here we apply first fully connected layer with 100 outputs
Here is the first time when we introduce Dropout with Dropout rate as 0.25 to combact overfitting
Next we introduce second fully connected layer with 50 outputs
Then comes a third connected layer with 10 outputs
And finally the layer with one output.
Here we require one output just because this is a regression problem and we need to predict the steering angle.
####Attempts to reduce overfitting in the model
After the full connected layer I have used a dropout so that the model generalizes on a
 track that it has not seen. I decided to keep the Dropoout rate as 0.25 to combact 
overfitting. model using a Keras lambda layer.
#####Model parameter tuning
No of epochs= 5
Optimizer Used- Adam
Learning Rate- Default 0.001
Validation Data split- 0.15
Generator batch size= 32
Correction factor- 0.2
Loss Function Used- MSE(Mean Squared Error as it is efficient for regression problem).




