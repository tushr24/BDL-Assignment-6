# BDL Assignment 6

# Objective
Understand and implement FastAPI for handwritten digit classification. 
FastAPI is an open-source web framework for creating APIs with Python that's used for data science and e-commerce applications.

# Purpose
This project aims to build a FastAPI application that exposes the functionality of an MNIST digit classification model over a RESTful API. 
Users can upload images of handwritten digits, and the API will predict the digit based on the trained model.

# Dataset
- Task 1 images were downloaded from the attached hugging face [link](https://huggingface.co/datasets/mnist).
- Task 2 images were created by taking screenshots of hand-drawn digits on a touch screen.

# Steps to execute the API
- Open the command line terminal in the working directory containing all the Python files and the model.
- For task 1, type `python CH20B025_Task1.py mnist-epoch-10.keras`. 
- For task 2, type `python CH20B025_Task2.py mnist-epoch-10.keras`.
- Then open your web browser and type `http://localhost:8000/docs`.
- Based on the chosen task, upload the acceptable images.

# Brief Description
This project is a FastAPI application that loads a pre-trained MNIST digit classification model. 
The application provides an endpoint (/predict) where users can upload an image of a handwritten digit. 
The image is preprocessed, and the model predicts the digit based on the processed image data. 
The predicted digit is then returned to the user as a JSON response.

- For `CH20B025_Task1.py`, upload images of size *28 x 28*.
- For `CH20B025_Task2.py`, any image with a single digit will work.

# Conclusions
The API can classify the digit in images (PNG/JPEG) that are uploaded by the user. 

- The model performed well on images for task 1 since those resembled the images on which the model had been trained.
- Since the model was built on the MNIST dataset, which has images with white text and black backgrounds, it could not correctly identify the **digit** image.
- The model could also not accurately predict the images **digit-4** & **digit-7** for task 2.

