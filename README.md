[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!

## Import Datsets

1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). 
2. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).   

## CNN Model

1. Layer 1 comprises of Sequential operation of 1-Conv2D layer(3,32,3,2,1), 1-RELU and 1-MaxPool2D
2. Layer 2 comprises of Sequential operation of 1-Conv2D layer(32,64,3,2,1), 1-RELU and 1-MaxPool2D
3. Layer 3 comprises of Sequential operation of 1-Conv2D layer(64,128,3,1,1), 1-RELU and 1-MaxPool2D
4. Layer 4 comprises of Sequential operation of Linear(6272,500) and Dropout(0.3)
5. Final Linear layer mapped to 133 outputs

## Transfer Learning

For the Transfer learning, Resnet50 is used. 

## Packages used

1. torch, torchvision
2. numpy

## Files
1. dog_app.html - html file of the code
2. dog_app.ipynb - Jupyter notebook file of the classifier
3. model.py - CNN net
4. /test_images - user defined test images

## Results & Conclusion

Resnet50 model using transfer learning was able to achieve 85% accuracy on the test data. An algorithm was defined to classify images as a particular dog breed and in case of human, their closest resemblances. It worked almost all the time.  

There are possible improvements suggested in the notebook like image augmentation, ensemble models etc.

## Report
Technical report is available in [medium](https://medium.com/@srinivasanvasu/udacity-data-science-capstone-project-dog-breed-classifier-e7539718c8a2)

## References
1. Pytorch Documentation in developing CNN and using Resnet50.
2. Teddylee777 — Github repository is referred for basic structure and report. 