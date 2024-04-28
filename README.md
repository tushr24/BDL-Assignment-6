# BDL Assignment 6

# Objective
Get to know how to use FastAPI and implement FastAPI for handwritten digit classification. 
This project builds a FastAPI application that shows the mnist digit classification model over a RESTful API.
Here anyone can upload images of digits handwritten or found from anywhere and the API will try to predict the digit based on the model trained.

# Dataset
- Task 1 images were downloaded from the attached hugging face [link](https://huggingface.co/datasets/mnist).
- Task 2 images were created by handwriting digits on a black background on a mobile phone.

# Steps to execute the API
- Open the command prompt and go to the directory where all the files are stored.
- For task 1, type `python CH20B025_Task1.py mnist-epoch-10.keras`.
- Then open your web browser and type `http://localhost:8000/docs`.
- Upload 28*28 images for task 1. 
- For task 2, type `python CH20B025_Task2.py mnist-epoch-10.keras`.
- Then open your web browser and type `http://localhost:8000/docs`.
- For this you can upload any size image.

# Brief Description
This project is a FastAPI application that loads a pre-trained MNIST digit classification model. 
The application provides an endpoint (/predict) where users can upload an image of a handwritten digit. 
The image is preprocessed, and the model predicts the digit based on the processed image data. 
The predicted digit is then returned to the user as a JSON response.

- For `CH20B025_Task1.py`, upload images of size *28 x 28*.
- For `CH20B025_Task2.py`, any image with a single digit will work.

# Conclusions
The API can classify the digit in images (PNG/JPEG) that are uploaded by the user. 

- The model performs well on task 1 images since those were the images the model was trained on.
- Since the model was built on the MNIST dataset, which has images with white text and black backgrounds, it could not correctly identify all the task 2 images uploaded.
- The model could also not accurately predict the images **digit-1**, **digit-4** & **digit-8** for task 2.

