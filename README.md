# Real-Time-Sign-Language-Recognition
Real time detection of American Sign Language Letters (Hand Gestures) via CNN and OpenCV

This project helps you to understand the concepts of Computer Vision, CNN (Convolutional Neural Network) Model and combining them to work together.

<h2> 1. Importing Data And Libraries </h2>
<p> In 'Hand Gesture Recognition.py' file first and foremost, we start by importing our required libraries then import our datasets.( You can find the datasets in this link: https://www.kaggle.com/ekremzturk/sign-language-recognition/data) </p>

<h2> 2. Preprocessing </h2>
<p> After importing our dataset, we create subsets of train and test data. Afterwards, X_train and X_test data are reshaped in order to fit the CNN Model. Because CNN Model takes dataset shaped like (numofimages,xpixels,ypixels,channelofimgs). In order to reduce the work load of the model we normalized the images by dividing their pixel values by max pixel value in this case max pixel value is 255.</p>

<h2> 3. Creating and Training The CNN Model </h2>
<p> To create a CNN Model we start with assigning the Sequential function to model variable. Then we add layers one by one. At last we compile it with 'sparse_categorical_crossentropy' loss because our target values (y_train and y_test) are not one-hot encoded.</p>

<h2> 4. Saving The Model</h2>
<p> In order to save the hist assigned model we use pickle library. And save it as a pickle file.( You can name it whatever you want) </p>

<h2> 5. Real Time Detection </h2>
<p> After importing required libraries we load our pickle file CNN Model and then use the model to predict the input values. In this case our input value is cropped rectangle of the video screen. We choose a part of the screen because we trained our model with hand images so when we place our hand in the indicated area our model gives us a prediction then print it on the screen.</p>

<p> I hope you find this project helpful</p>


  
  
