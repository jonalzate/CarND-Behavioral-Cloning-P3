# **Behavioral Cloning Project** 

---

[image1]: ./report_img/title_image.png "Behavioral Cloning"
[image2]: ./model_summary.png "Model Summary"
[image3]: ./model_summary_revised.png "Revised Model Summary"
[image4]: ./report_img/left.png "Left Image"
[image5]: ./report_img/right.png "Right Image"
[image6]: ./report_img/center.png "Center Image"
[image7]: ./report_img/loss_plot.png "Left Image"


![image1]




The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report







## Rubric Points 

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I decided to implement the End-to-End model from NVIDIA because it seemed to be the most adequate for the task, in my personal opinion. The model has the following architecture, from model.summary()



#### 2. Attempts to reduce overfitting in the model
 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 28). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
To reduce overfitting the model implements couple of Dropout layers after each convolution layer (lines 35-53)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 51).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road on both tracks
to generalize the data. 

For details about how I created the training data, see the next section. 



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to choose from available models learned or referenced in class the model that seem best suited for the task at hand. The NVIDIA
model definitely seem the better choice, it included convolutional layers and different activation functions so I decided to go with this model architecture.

Considering computation limitation I decided to have as much processing happening on the GPU so the first layer of my model was replaced with a Cropping2D layer to remove trees and skyline from
images from the top and the hood of the car from the bottom. The second layer is a Lambda layer that takes care of the normalization step. Another step to reduce computation on CPU was to use 
the Generator class from Keras in order to have all the image preprocessing (image flipping, shifting, color change, etc) happening on the GPU.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track may be due to left/right turn bias.
To improve the driving behavior in these cases, I recorded a couple of training runs from both tracks and used images from all three cameras to have the model trained in more data

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 31-46) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

*Model Architecture*

![image2]

**Revised model with Dropout layers**

*Revised Model Architecture*

![image3]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps in both directions (clockwise, counterclockwise) on track one using center lane driving. Here is an example image of center lane driving:

*Left View*

![image4]

*Center View*

![image5]

*Right View*

![image6]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to adjust itself back on to the center

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would compensate for any left/right turn bias, when not using all three cameras.
I also gathered driving data from track two, three laps in each direction clockwise and counterclockwise to compensate any turn bias. 

After the collection process, I then preprocessed this data by doing the following:

* flipping
* convert to rgb
* crop
* normalized


I finally randomly shuffled the data set and split data into a training and validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 7-10 as evidenced by the loss and validation loss not having any improvements after 10 epochs
I used an adam optimizer so that manually training the learning rate wasn't necessary, but I still left the option open to set a tunable learning rate.

### Loss Plot

The loss and validation loss plot after training 

![image7]
