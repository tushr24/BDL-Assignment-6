import sys
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from keras.models import Sequential
from keras.models import load_model as keras_model


app = FastAPI()

# Take the path of the model as a command line argument
if len(sys.argv) < 2:
    raise ValueError("Please provide the path to the model file as a command-line argument.")
model_path = sys.argv[1]

# Create a function "def load_model(path:str) -> Sequential" which will load the model saved at the supplied path on the disk and return the keras.src.engine.sequential.Sequential model.
def load_model(path: str) -> Sequential:
    """
    Load the pre-trained Keras model from the specified path.
    
    Args:
        path (str): The path to the saved model file.
        
    Returns:
        Sequential: The loaded Keras Sequential model.
    """
    return keras_model(path)

# Load the model
model = load_model(model_path)

# Create a function "def predict_digit(model:Sequential, data_point:list) -> str" that will take the image serialized as an array of 784 elements and returns the predicted digit as string.
def predict_digit(model: Sequential, image: np.ndarray) -> str:
    """
    Predict the digit in the given image using the loaded model.
    
    Args:
        model (Sequential): The loaded Keras Sequential model.
        image (np.ndarray): The input image as a NumPy array.
        
    Returns:
        str: The predicted digit as a string.
    """
    # Preprocess the image
    image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = image.reshape(1, 784) / 255.0  # Reshape and normalize pixel values

    # Make prediction
    prediction = model.predict(image)
    digit = str(np.argmax(prediction))
    return digit

# Create an API endpoint "@app.post('/predict')" that will read the bytes from the uploaded image to create a serialized array of 784 elements. The array shall be sent to 'predict_digit' function to get the digit. The API endpoint should return {"digit":digit"} back to the client.
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    API endpoint to predict the digit in an uploaded image.
    
    Args:
        file (UploadFile): The uploaded image file.
        
    Returns:
        dict: A dictionary containing the predicted digit.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    digit = predict_digit(model, image)
    return {"digit": digit}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)

